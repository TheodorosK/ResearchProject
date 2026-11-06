"""Neuroevolution benchmark: a genetic algorithm evolving Keras MLPs on a synthetic regression task,
plus an isolated memory-growth diagnostic.

Protocol (scientific-validity rules)
------------------------------------
* Fitness used for selection = validation R^2.  Training R^2 is recorded separately.
* The test split is touched exactly ONCE, after evolution, for the single best-by-validation individual.
* Every individual gets the same fixed budget: ``ne_epochs`` passes over ``ne_train_n`` samples with batch
  ``ne_batch``; per-individual seeds derive from (seed, uid); initialisers are explicitly seeded.
* Elites are carried over unchanged and not re-evaluated.
* With --ne-inherit-weights children continue from (perturbed) parent weights, so their effective training
  budget is CUMULATIVE; fitness curves are then not comparable to a no-inheritance run (recorded in the report).
"""
from __future__ import annotations

import gc
import math
import random
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import genetics as G
from . import tfenv
from .common import Recorder, classify_memory_growth, timing_stats
from .limits import LimitExceeded, ResourceMonitor, extra_gb, take_sample
from .sysinfo import CpuMeter, memory_pressure_free_pct, peak_rss_gb, swap_counters

pc = time.perf_counter


# ----------------------------------------------------------------------- data
def make_dataset(n_train: int, n_val: int, n_test: int, in_dim: int, seed: int) -> Dict[str, np.ndarray]:
    """Reproducible nonlinear regression task with observation noise (std-normalised by train stats)."""
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    x = rng.uniform(-1, 1, (n, in_dim)).astype(np.float32)
    a = rng.normal(0, 1 / math.sqrt(in_dim), in_dim)
    b = rng.normal(0, 1 / math.sqrt(in_dim), in_dim)
    y = (np.sin(3.0 * x @ a) + 0.5 * x[:, 0] * x[:, 1 % in_dim] + 0.25 * np.tanh(2.0 * x @ b) ** 2
         + rng.normal(0, 0.1, n))
    mu, sd = y[:n_train].mean(), y[:n_train].std() or 1.0
    y = ((y - mu) / sd).astype(np.float32)
    s1, s2 = n_train, n_train + n_val
    return {"x_train": x[:s1], "y_train": y[:s1], "x_val": x[s1:s2], "y_val": y[s1:s2],
            "x_test": x[s2:], "y_test": y[s2:]}


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    sst = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return 1.0 - float(np.sum((y - pred) ** 2)) / sst


# ----------------------------------------------------------------- evaluation
@dataclass
class EvalResult:
    uid: int
    status: str = "ok"                 # ok | failed
    error: str = ""
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    build_s: float = 0.0
    train_s: float = 0.0               # model.fit wall time (includes per-model tracing/compilation)
    eval_s: float = 0.0
    total_s: float = 0.0
    params: int = 0
    inherited_fraction: float = 0.0
    loss_trace: Optional[List[float]] = None
    diag: Optional[Dict[str, Any]] = None
    weights: Optional[List[np.ndarray]] = field(default=None, repr=False)


@dataclass
class Individual:
    genome: G.Genome
    result: Optional[EvalResult] = None
    donor: Optional["Individual"] = None


def build_keras(tf, genome: G.Genome, in_dim: int, seed: int):
    """Zeros-initialised layers + host-generated Glorot weights (identical on CPU and GPU; avoids the
    random-number op that tensorflow-metal cannot run on the GPU)."""
    from .workloads import glorot_init
    K = tf.keras.layers
    layers = [K.Input((in_dim,))]
    layers += [K.Dense(l.units, activation=l.activation, kernel_initializer="zeros") for l in genome.layers]
    layers.append(K.Dense(1, kernel_initializer="zeros"))
    model = tf.keras.Sequential(layers)
    model.set_weights(glorot_init([w.shape for w in model.get_weights()], seed))
    return model


def train_steps(args, n_train: int) -> int:
    """Fixed per-individual budget: epochs x floor(train_n / batch) optimiser steps."""
    return args.ne_epochs * max(1, n_train // args.ne_batch)


def evaluate(tf, device: str, genome: G.Genome, data: Dict[str, np.ndarray], args,
             donor: Optional[EvalResult] = None, donor_depth: int = 0, trace_steps: int = 0) -> EvalResult:
    res = EvalResult(uid=genome.uid, params=G.count_params(genome, data["x_train"].shape[1], 1))
    seed = (args.seed * 1_000_003 + genome.uid) % (2 ** 31 - 1)
    t_all = pc()
    model = None
    try:
        with tfenv.device_scope(tf, device):
            t = pc()
            model = build_keras(tf, genome, data["x_train"].shape[1], seed)
            if donor is not None and donor.weights is not None:
                w, res.inherited_fraction = G.inherit_weights(
                    donor.weights, donor_depth, model.get_weights(), len(genome.layers))
                w = G.perturb_weights(w, np.random.default_rng(seed), genome.weight_sigma, genome.weight_rate)
                model.set_weights(w)
            res.build_s = pc() - t

            n = len(data["x_train"])
            bs = args.ne_batch
            nb = max(1, n // bs)
            rng = np.random.default_rng(seed)
            idx = np.concatenate([rng.permutation(n)[: nb * bs] for _ in range(args.ne_epochs)])
            xe = tf.constant(data["x_train"][idx].reshape(-1, bs, data["x_train"].shape[1]))
            ye = tf.constant(data["y_train"][idx].reshape(-1, bs))
            tvars = model.trainable_variables
            opt, _ = tfenv.make_optimizer(tf, genome.learning_rate, tvars, getattr(args, "optimizer", "simple"))

            @tf.function(jit_compile=False)
            def step(x, y):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.square(y - tf.squeeze(model(x, training=True), -1)))
                opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
                return loss

            if trace_steps:                                  # forward-pass diagnosis before ANY update
                from .workloads import host_forward, host_mse, weights_checksum
                w0 = model.get_weights()
                x0, y0 = xe[0].numpy(), ye[0].numpy()
                mse = lambda x, y: tf.reduce_mean(tf.square(y - tf.squeeze(model(x, training=False), -1)))
                host = host_forward(w0, x0, [l.activation for l in genome.layers])
                res.diag = {
                    "init_weights_checksum": weights_checksum(w0),
                    "batch0_checksum": float(np.abs(x0.astype(np.float64)).sum()),
                    "eager_forward_loss": float(mse(xe[0], ye[0]).numpy()),
                    "graph_forward_loss": float(tf.function(mse, jit_compile=False)(xe[0], ye[0]).numpy()),
                    "host_truth_loss": host_mse(host, y0) if host is not None else None}
            t = pc()
            loss, trace = None, []
            for i in range(xe.shape[0]):
                loss = step(xe[i], ye[i])
                if i < trace_steps:                       # diagnostic only (syncs each traced step)
                    trace.append(float(loss.numpy()))
            res.loss_trace = trace or None
            tfenv.sync(loss, tvars[-1])                      # completion barrier before stopping the clock
            res.train_s = pc() - t

            t = pc()
            p_tr = model(tf.constant(data["x_train"]), training=False).numpy()
            p_va = model(tf.constant(data["x_val"]), training=False).numpy()
            res.eval_s = pc() - t
        res.train_r2, res.val_r2 = r2_score(data["y_train"], p_tr), r2_score(data["y_val"], p_va)
        if not (math.isfinite(res.train_r2) and math.isfinite(res.val_r2)):
            raise FloatingPointError("non-finite fitness (diverged)")
        res.weights = model.get_weights()
    except LimitExceeded:
        raise
    except Exception as e:                       # a failed evaluation is data, not a crash
        res.status, res.error = "failed", f"{type(e).__name__}: {str(e)[:200]}"
        res.train_r2 = res.val_r2 = None
        if tfenv.is_placement_error(e):
            raise
    finally:
        res.total_s = pc() - t_all
        del model
    return res


def _cleanup(tf, clear: bool) -> float:
    t = pc()
    if clear:
        tf.keras.backend.clear_session()
    gc.collect()
    return pc() - t


# ------------------------------------------------------------------ evolution
def run(args, rec: Recorder, monitor: ResourceMonitor, device: str, placement: str) -> None:
    tf = tfenv.configure(device, placement, args.grappler)
    rec.data["tf"] = tfenv.tf_info(tf)
    ctx = dict(device=device, placement=placement)
    units = tuple(u for u in (8, 16, 32, 64, 128, 256, 512) if u <= args.ne_max_units)
    space = G.SearchSpace(max_layers=args.ne_max_layers, unit_choices=units,
                          activations=tuple(args.ne_activations))
    data = make_dataset(args.ne_train_n, args.ne_val_n, args.ne_test_n, args.ne_in_dim, args.seed)
    rng = random.Random(args.seed)
    workers = max(1, args.ne_workers)
    policy = args.ne_cleanup
    if workers > 1 and policy == "each":
        policy = "generation"
        rec.note("--ne-cleanup each is unsafe with parallel evaluation (clear_session is process-global); "
                 "using 'generation' instead.")
    rec.data["config"] = {
        "device": device, "placement": placement, "population": args.ne_pop, "generations": args.ne_generations,
        "elite": args.ne_elite, "tournament": args.ne_tournament, "crossover_rate": args.ne_crossover,
        "epochs": args.ne_epochs, "batch": args.ne_batch, "train_n": args.ne_train_n, "val_n": args.ne_val_n,
        "test_n": args.ne_test_n, "in_dim": args.ne_in_dim, "workers": workers, "cleanup_policy": policy,
        "inherit_weights": bool(args.ne_inherit_weights), "seed": args.seed,
        "optimizer": getattr(args, "optimizer", "simple"), "grappler": args.grappler,
        "search_space": {"layers": [space.min_layers, space.max_layers], "units": list(space.unit_choices),
                         "activations": list(space.activations), "lr": list(space.lr_range)},
    }
    if args.ne_inherit_weights:
        rec.note("Weight inheritance is ON: children continue training from parent weights, so the effective "
                 "training budget is cumulative and fitness is not comparable with a non-inheriting run.")
    if workers > 1:
        rec.note("Parallel evaluation uses threads sharing one TensorFlow runtime (and one Metal GPU on "
                 "device=gpu); speed-up is not assumed and is reported as measured.")
    rec.note("Training time per individual includes per-model graph tracing/compilation (realistic for "
             "neuroevolution, where every individual is a new model).")

    # ---- verify placement on a probe model (never assume the GPU is used) ------------------
    with tfenv.device_scope(tf, device):
        probe = build_keras(tf, G.Genome([G.LayerGene(8, "relu")], 1e-3), args.ne_in_dim, 0)
    ev = tfenv.placement_evidence(tf, device, probe.trainable_variables, {})
    ev["optimizer_class"] = tfenv.make_optimizer(tf, 1e-3, [], getattr(args, "optimizer", "simple"))[1]
    rec.data["device_evidence"] = ev
    del probe
    if not ev["verified"]:
        rec.set_status("failed", "device placement could not be verified: " + "; ".join(ev["problems"]))
        return

    # ---- fixed reference genome: same genome/seed/data on every device => comparable fitness -------
    ref_genome = G.Genome([G.LayerGene(128, "relu"), G.LayerGene(128, "relu"), G.LayerGene(64, "tanh")],
                          learning_rate=1e-3, uid=-1)
    t = pc()
    ref = evaluate(tf, device, ref_genome, data, args, trace_steps=10)
    rec.data["reference_check"] = {"genome": ref_genome.describe(), "status": ref.status, "val_r2": ref.val_r2,
                                   "train_r2": ref.train_r2, "total_s": pc() - t, "error": ref.error,
                                   "loss_trace": ref.loss_trace, "diag": ref.diag, "optimizer": getattr(args, "optimizer", "simple")}
    rec.measure("reference_val_r2", ref.val_r2, "R2", **ctx)
    rec.measure("reference_train_r2", ref.train_r2, "R2", **ctx)
    rec.measure("reference_eval_cold_start_s", pc() - t, "s", **ctx)
    rec.note("A fixed reference genome is evaluated once before generation 0 (not counted in the evaluation "
             "totals). It absorbs one-time start-up cost (runtime/Metal initialisation, first compilation) and its "
             "val R2 is compared across devices to detect numerical disagreement.")
    _cleanup(tf, True)
    from .workloads import numeric_gate
    gate = None if args.skip_numeric_gate else numeric_gate(ref.diag)
    if gate:
        rec.set_status("failed", gate)
        rec.note("This is a numerical-correctness failure of the device/graph execution, not a benchmark failure. "
                 "Run tools/metal_numerics_check.py; on tensorflow-metal, `--grappler none` (disable all graph rewriting, applied to CPU and GPU jobs alike) is the known workaround.")
        return

    uid = [0]

    def new_uid() -> int:
        uid[0] += 1
        return uid[0] - 1

    pop = [Individual(G.random_genome(rng, space, uid=new_uid())) for _ in range(args.ne_pop)]
    uid[0] = args.ne_pop
    cpu = CpuMeter()
    best_overall: Optional[Individual] = None
    gen_records: List[Dict[str, Any]] = []

    def eval_one(ind: Individual) -> EvalResult:
        monitor.raise_if_aborted()
        donor = ind.donor.result if (ind.donor is not None and args.ne_inherit_weights) else None
        depth = len(ind.donor.genome.layers) if donor is not None else 0
        r = evaluate(tf, device, ind.genome, data, args, donor, depth)
        if policy == "each":
            tf.keras.backend.clear_session()
            gc.collect()
        return r

    def eval_many(inds: List[Individual], n_workers: int) -> List[EvalResult]:
        if n_workers <= 1:
            return [eval_one(i) for i in inds]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(eval_one, inds))

    # ---- optional controlled parallel-vs-sequential probe (results discarded) ----
    if workers > 1:
        k = min(len(pop), 2 * workers)
        sample = pop[:k]
        t = pc(); seq = eval_many(sample, 1); t_seq = pc() - t
        _cleanup(tf, True)
        t = pc(); par = eval_many(sample, workers); t_par = pc() - t
        _cleanup(tf, True)
        rec.data["parallel_probe"] = {"individuals": k, "workers": workers, "sequential_s": t_seq,
                                      "parallel_s": t_par, "speedup": t_seq / t_par if t_par else None,
                                      "failures": sum(r.status != "ok" for r in seq + par)}
        rec.measure("parallel_probe_speedup", t_seq / t_par if t_par else float("nan"), "x", kind="calculated",
                    workers=workers, **ctx)
        rec.measure("parallel_probe_sequential_ind_per_s", k / t_seq, "ind/s", kind="calculated", **ctx)
        rec.measure("parallel_probe_parallel_ind_per_s", k / t_par, "ind/s", kind="calculated", **ctx)
        rec.checkpoint()

    mp_before = memory_pressure_free_pct()
    t_run = pc()
    total_eval = total_fail = 0
    monitor.window_peaks()                       # reset
    for gen in range(args.ne_generations):
        monitor.raise_if_aborted()
        monitor.window_peaks()
        cpu.reset()
        swap0 = swap_counters()
        mem0 = take_sample()
        todo = [i for i in pop if i.result is None]
        t0 = pc()
        results = eval_many(todo, workers)
        wall = pc() - t0
        for ind, r in zip(todo, results):
            ind.result = r
        ok = [r for r in results if r.status == "ok"]
        failed = [r for r in results if r.status != "ok"]
        total_eval += len(results)
        total_fail += len(failed)
        cpu_read = cpu.read()
        peaks = monitor.window_peaks()
        mem_before = take_sample()
        cleanup_s = _cleanup(tf, policy in ("generation", "each")) if policy != "never" else 0.0
        mem_after = take_sample()
        sw = swap_counters()

        scored = [i.result.val_r2 if i.result and i.result.status == "ok" else None for i in pop]
        best_i = max((i for i in range(len(pop)) if scored[i] is not None), key=lambda i: scored[i], default=None)
        if best_i is not None and (best_overall is None or scored[best_i] > best_overall.result.val_r2):
            best_overall = pop[best_i]
        vals = [r.val_r2 for r in ok]
        rec_g = {
            "generation": gen, "evaluated": len(results), "ok": len(ok), "failed": len(failed),
            "elites_carried": len(pop) - len(todo),
            "errors": sorted({r.error for r in failed})[:3],
            "wall_s": wall, "individuals_per_s": (len(results) / wall) if wall else None,
            "train_s": timing_stats([r.train_s for r in ok]),
            "build_s_mean": float(np.mean([r.build_s for r in ok])) if ok else None,
            "best_val_r2": max(vals) if vals else None,
            "mean_val_r2": float(np.mean(vals)) if vals else None,
            "best_train_r2": max(r.train_r2 for r in ok) if ok else None,
            "best_so_far_val_r2": best_overall.result.val_r2 if best_overall else None,
            "mean_params": float(np.mean([r.params for r in ok])) if ok else None,
            "process_cores_busy": cpu_read["process_cores_busy"], "system_cpu_pct": cpu_read["system_cpu_pct"],
            "rss_gb_before_cleanup": mem_before.rss_gb, "rss_gb_after_cleanup": mem_after.rss_gb,
            "extra_system_gb_before_cleanup": extra_gb(monitor.baseline, mem_before),
            "extra_system_gb_after_cleanup": extra_gb(monitor.baseline, mem_after),
            "available_gb_after_cleanup": mem_after.avail_gb,
            "peak_extra_system_gb": peaks["max_extra_gb"], "peak_rss_gb": peaks["max_rss_gb"],
            "min_available_gb": peaks["min_avail_gb"],
            "swap_growth_mb": (sw["used_gb"] - swap0["used_gb"]) * 1024,
            "swap_growth_since_start_mb": (sw["used_gb"] - monitor.baseline.swap_used_gb) * 1024,
            "cleanup_s": cleanup_s,
        }
        gen_records.append(rec_g)
        rec.data["generations"] = gen_records
        for key, unit in (("wall_s", "s"), ("individuals_per_s", "ind/s"), ("best_val_r2", "R2"),
                          ("mean_val_r2", "R2"), ("peak_extra_system_gb", "GB"), ("rss_gb_after_cleanup", "GB"),
                          ("extra_system_gb_after_cleanup", "GB"), ("swap_growth_mb", "MB"),
                          ("process_cores_busy", "cores"), ("cleanup_s", "s")):
            if rec_g.get(key) is not None:
                rec.measure(f"gen_{key}", rec_g[key], unit,
                            kind="calculated" if key in ("individuals_per_s",) else "measured",
                            generation=gen, **ctx)
        rec.measure("gen_failed", len(failed), "count", generation=gen, **ctx)
        rec.checkpoint()
        print(f"  gen {gen}: {len(results)} evals ({len(failed)} failed) in {wall:.1f}s "
              f"| best val R2 {rec_g['best_val_r2']} | extra mem {rec_g['extra_system_gb_after_cleanup']:.2f} GB "
              f"| swap +{rec_g['swap_growth_since_start_mb']:.0f} MB", flush=True)

        if gen < args.ne_generations - 1:
            offspring, uid[0] = G.next_generation(
                [i.genome for i in pop], scored, rng, space, elite_k=args.ne_elite,
                tournament_k=args.ne_tournament, crossover_rate=args.ne_crossover,
                next_uid=uid[0], generation=gen + 1)
            new_pop = []
            for o in offspring:
                if o.is_elite:
                    new_pop.append(pop[o.primary_parent])       # same object: result retained, no re-eval
                else:
                    donor = pop[o.primary_parent] if args.ne_inherit_weights else None
                    new_pop.append(Individual(o.genome, None, donor))
            pop = new_pop

    # ---- summary -------------------------------------------------------------
    total_wall = pc() - t_run
    rec.measure("total_evaluations", total_eval, "count", **ctx)
    rec.measure("failed_evaluations", total_fail, "count", **ctx)
    rec.measure("successful_evaluations", total_eval - total_fail, "count", **ctx)
    rec.measure("overall_individuals_per_s", total_eval / total_wall if total_wall else float("nan"),
                "ind/s", kind="calculated", **ctx)
    rec.measure("total_evolution_s", total_wall, "s", **ctx)
    if gen_records and gen_records[0]["best_val_r2"] is not None and gen_records[-1]["best_so_far_val_r2"] is not None:
        rec.measure("val_r2_improvement_first_to_last_generation",
                    gen_records[-1]["best_so_far_val_r2"] - gen_records[0]["best_val_r2"], "R2",
                    kind="calculated", **ctx)
    rec.measure("peak_rss_gb", peak_rss_gb(), "GB", **ctx)
    pk = monitor.run_peaks
    rec.measure("peak_extra_system_gb_sampled", pk["max_extra_gb"], "GB", **ctx)
    rec.measure("min_available_gb_sampled", pk["min_avail_gb"], "GB", **ctx)
    rec.measure("memory_free_pct_before_macos", mp_before, "%", **ctx)
    rec.measure("memory_free_pct_after_macos", memory_pressure_free_pct(), "%", **ctx)
    growth = [g["extra_system_gb_after_cleanup"] for g in gen_records]
    if len(growth) >= 2:
        rec.measure("system_memory_growth_per_generation_mb",
                    (growth[-1] - growth[0]) / (len(growth) - 1) * 1024, "MB/gen", kind="calculated", **ctx)
        rss = [g["rss_gb_after_cleanup"] for g in gen_records]
        rec.measure("rss_growth_per_generation_mb", (rss[-1] - rss[0]) / (len(rss) - 1) * 1024, "MB/gen",
                    kind="calculated", **ctx)
        rec.note("Per-generation growth is a coarse indicator only; use the `memdiag` test to distinguish "
                 "retained memory, allocator caching and leaks.")

    # ---- single, final test-set evaluation (never used for selection) -----------
    if best_overall is not None and best_overall.result.weights is not None:
        with tfenv.device_scope(tf, device):
            m = build_keras(tf, best_overall.genome, args.ne_in_dim, 0)
            m.set_weights(best_overall.result.weights)
            pred = m(tf.constant(data["x_test"]), training=False).numpy()
        test_r2 = r2_score(data["y_test"], pred)
        rec.measure("best_individual_val_r2", best_overall.result.val_r2, "R2", **ctx)
        rec.measure("best_individual_train_r2", best_overall.result.train_r2, "R2", **ctx)
        rec.measure("best_individual_test_r2", test_r2, "R2", **ctx)
        rec.data["best_individual"] = {
            "architecture": best_overall.genome.describe(), "uid": best_overall.genome.uid,
            "generation": best_overall.genome.generation, "params": best_overall.result.params,
            "val_r2": best_overall.result.val_r2, "train_r2": best_overall.result.train_r2,
            "test_r2": test_r2, "test_set_used_for_selection": False}
    rec.data["evaluation_failures"] = {"count": total_fail}
    rec.note("Test R2 was computed once, after evolution, for the best-by-validation individual only.")


# ------------------------------------------------------------ memory diagnostic
def run_memdiag(args, rec: Recorder, monitor: ResourceMonitor, device: str, placement: str, policy: str) -> None:
    """Build -> fit -> discard the SAME model N times under one cleanup policy, in a fresh process.

    policy: none  (drop references only; interpreter GC runs when it wants)
            gc    (gc.collect() after each cycle)
            clear (tf.keras.backend.clear_session() + gc.collect() after each cycle)
    """
    tf = tfenv.configure(device, placement, args.grappler)
    ctx = dict(device=device, placement=placement, policy=policy)
    rec.data["tf"] = tfenv.tf_info(tf)
    genome = G.Genome([G.LayerGene(128, "relu"), G.LayerGene(128, "relu"), G.LayerGene(64, "tanh")],
                      learning_rate=1e-3, uid=0)
    data = make_dataset(args.ne_train_n, args.ne_val_n, 64, args.ne_in_dim, args.seed)
    cycles = args.memdiag_cycles
    rec.data["config"] = {"policy": policy, "cycles": cycles, "genome": genome.describe(), "device": device,
                          "seed": args.seed, "warmup_cycles": 3}
    rss, sysx, py, ngc = [], [], [], []
    tracemalloc.start()
    try:
        for c in range(cycles):
            monitor.raise_if_aborted()
            r = evaluate(tf, device, genome, data, args)
            del r
            if policy == "gc":
                gc.collect()
            elif policy == "clear":
                tf.keras.backend.clear_session()
                gc.collect()
            s = take_sample()
            cur, _ = tracemalloc.get_traced_memory()
            rss.append(s.rss_gb * 1024)
            sysx.append(extra_gb(monitor.baseline, s) * 1024)
            py.append(cur / 1024 ** 2)
            ngc.append(len(gc.get_objects()))
            rec.measure("cycle_rss_mb", rss[-1], "MB", cycle=c, **ctx)
            rec.measure("cycle_extra_system_mb", sysx[-1], "MB", cycle=c, **ctx)
            rec.measure("cycle_python_traced_mb", py[-1], "MB", cycle=c, **ctx)
            rec.measure("cycle_gc_objects", ngc[-1], "count", cycle=c, **ctx)
            rec.checkpoint()
    finally:
        tracemalloc.stop()
    c_rss = classify_memory_growth(rss, py, warmup=3, slope_threshold_mb=args.memdiag_threshold_mb)
    c_sys = classify_memory_growth(sysx, py, warmup=3, slope_threshold_mb=args.memdiag_threshold_mb)
    rec.data["classification"] = {"rss": c_rss, "system_extra": c_sys}
    for name, c in (("rss", c_rss), ("system_extra", c_sys)):
        rec.measure(f"growth_verdict_{name}", c["verdict"], "", kind="calculated", **ctx)
        if "slope_mb_per_cycle" in c:
            rec.measure(f"growth_slope_{name}_mb_per_cycle", c["slope_mb_per_cycle"], "MB/cycle",
                        kind="calculated", **ctx)
    obj_slope = None
    if len(ngc) > 6:
        from .common import linear_slope
        obj_slope = linear_slope(ngc[3:])
        rec.measure("gc_objects_slope_per_cycle", obj_slope, "objects/cycle", kind="calculated", **ctx)
    rec.note("Classification is heuristic: a leak is suspected only when growth continues at a similar rate "
             "across the whole post-warm-up window; rising RSS alone is not evidence of a leak.")
    rec.note("RSS excludes Metal/unified-memory buffers, so `system_extra` (drop in available memory) is "
             "reported alongside it; that series is noisy while other applications run.")
