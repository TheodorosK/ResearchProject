"""Execution of scaling cells: sequential and bucketed-batched evaluation, timing decomposition, cell/block runners.

Per-individual timing components (seconds, each closed by a host read-back barrier):
  construct   build model / stack weights, create optimiser, upload training batches
  trace       tf.function graph tracing (no execution)
  first_exec  first call: graph optimisation, kernel / shader compilation, dispatch warm-up, and training step #1
  train       training steps 2..S  (0 when S == 1)
  fitness     validation forward pass + host-side R2
  teardown    dropping the model / optimiser / function / device buffers at the end of the evaluation
plus, per cell: ga (selection / crossover / mutation), cleanup (gc.collect) and unattributed (loop overhead).
The training budget is exactly S optimiser steps per individual, so S=1 has no ``train`` component.
"""
from __future__ import annotations

import gc
import json
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .. import genetics as G
from .. import tfenv
from ..evolution import build_keras, r2_score
from ..limits import LimitExceeded, extra_gb, take_sample
from ..sysinfo import swap_counters
from ..workloads import glorot_init
from .workload import (Cell, Point, Settings, Strategy, batch_indices, bucket_genomes, dataset_for, elite_count,
                       individual_seed, initial_population, make_space, padding_overhead, params_of, workload_signature)

pc = time.perf_counter
COMP_KEYS = ("construct", "trace", "first_exec", "train", "fitness", "teardown")


class CellTimeout(Exception):
    """The per-configuration time limit was reached; the cell is recorded as aborted, never as a measurement."""


def _zero_comps() -> Dict[str, float]:
    return {k: 0.0 for k in COMP_KEYS}


# --------------------------------------------------------------------------------------- sequential
def eval_individual(tf, device: str, genome: G.Genome, data: Dict[str, np.ndarray], st: Settings, steps: int) -> Dict[str, Any]:
    seed = individual_seed(st.seed, genome.uid)
    comps = _zero_comps()
    res: Dict[str, Any] = {"uid": genome.uid, "status": "ok", "error": "", "val_r2": None, "comps": comps,
                           "params": params_of(genome, st), "batched": False, "bucket_size": 1}
    model = opt = step = xb = yb = tvars = loss = pred = None
    try:
        with tfenv.device_scope(tf, device):
            t = pc()
            model = build_keras(tf, genome, st.in_dim, seed)
            tvars = model.trainable_variables
            opt, _ = tfenv.make_optimizer(tf, genome.learning_rate, tvars, "simple")
            idx = batch_indices(seed, len(data["x_train"]), st.batch, steps)
            xb, yb = tf.constant(data["x_train"][idx]), tf.constant(data["y_train"][idx])
            tfenv.sync(tvars[-1], xb, yb)
            comps["construct"] = pc() - t

            @tf.function(jit_compile=False)
            def step(x, y):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.square(y - tf.squeeze(model(x, training=True), -1)))
                opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
                return loss

            t = pc()
            step.get_concrete_function(tf.TensorSpec((st.batch, st.in_dim), tf.float32), tf.TensorSpec((st.batch,), tf.float32))
            comps["trace"] = pc() - t
            t = pc()
            loss = step(xb[0], yb[0])
            tfenv.sync(loss, tvars[-1])
            comps["first_exec"] = pc() - t
            t = pc()
            for i in range(1, steps):
                loss = step(xb[i], yb[i])
            if steps > 1:
                tfenv.sync(loss, tvars[-1])
                comps["train"] = pc() - t
            t = pc()
            pred = model(tf.constant(data["x_val"]), training=False).numpy()
            comps["fitness"] = pc() - t
        r2 = r2_score(data["y_val"], pred)
        if not math.isfinite(r2):
            raise FloatingPointError("non-finite fitness (diverged)")
        res["val_r2"] = r2
    except LimitExceeded:
        raise
    except Exception as e:                                     # a diverged individual is data, a placement error is not
        if tfenv.is_placement_error(e):
            raise
        res["status"], res["error"] = "failed", f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        t = pc()
        model = opt = step = xb = yb = tvars = loss = pred = None       # release references (device buffers, traced function)
        comps["teardown"] = pc() - t
    return res


# ----------------------------------------------------------------------------------------- batched
class BatchedAdam:
    """Adam over population-stacked variables; per-individual learning rate; same update rule as SimpleAdam."""

    def __init__(self, tf, lr: np.ndarray, params, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        self.tf, self.b1, self.b2, self.eps = tf, beta_1, beta_2, epsilon
        self.lr = tf.constant(lr.astype(np.float32))                          # (P,1,1)
        self.m = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.v = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.t = tf.Variable(0.0, trainable=False)

    def apply_gradients(self, grads_and_vars):
        tf = self.tf
        self.t.assign_add(1.0)
        t = self.t.read_value()
        corr = tf.sqrt(1.0 - tf.pow(self.b2, t)) / (1.0 - tf.pow(self.b1, t))
        for (g, p), m, v in zip(grads_and_vars, self.m, self.v):
            m_new = self.b1 * m + (1.0 - self.b1) * g
            v_new = self.b2 * v + (1.0 - self.b2) * tf.square(g)
            m.assign(m_new)
            v.assign(v_new)
            p.assign_sub(self.lr * corr * m_new / (tf.sqrt(v_new) + self.eps))


def _activation(tf, name: str):
    return {"relu": tf.nn.relu, "tanh": tf.tanh, "sigmoid": tf.sigmoid, "swish": tf.nn.silu, "elu": tf.nn.elu}[name]


def eval_bucket_batched(tf, device: str, genomes: List[G.Genome], data: Dict[str, np.ndarray], st: Settings,
                        steps: int) -> List[Dict[str, Any]]:
    """Train a bucket of same-depth, same-activation genomes as ONE population-stacked network.

    Narrower members are zero-padded to the widest member and masked, so they are mathematically identical to separate
    training (padded weights stay exactly zero: zero input, masked output, zero gradient).  Padding costs extra FLOPs,
    reported as ``padded_ratio``.  Per-individual seeds, data order and learning rates match the sequential path.
    """
    P, depth = len(genomes), len(genomes[0].layers)
    acts = [l.activation for l in genomes[0].layers]
    widths = [max(g.layers[j].units for g in genomes) for j in range(depth)]
    in_d, out_d = [st.in_dim] + widths, widths + [1]
    seeds = [individual_seed(st.seed, g.uid) for g in genomes]
    comps = _zero_comps()
    n = len(data["x_train"])
    results = [{"uid": g.uid, "status": "ok", "error": "", "val_r2": None, "params": params_of(g, st), "batched": True,
                "bucket_size": P, "padded_ratio": padding_overhead(genomes, st.in_dim)} for g in genomes]
    Wv = bv = masks = tvars = opt = xbt = ybt = step = forward = loss = pred = None
    try:
        t = pc()
        W_np = [np.zeros((P, in_d[l], out_d[l]), np.float32) for l in range(depth + 1)]
        b_np = [np.zeros((P, 1, out_d[l]), np.float32) for l in range(depth + 1)]
        m_np = [np.zeros((P, 1, widths[j]), np.float32) for j in range(depth)]
        for i, g in enumerate(genomes):
            units, shapes, prev = [l.units for l in g.layers], [], st.in_dim
            for u in units:
                shapes += [(prev, u), (u,)]
                prev = u
            shapes += [(prev, 1), (1,)]
            w = glorot_init(shapes, seeds[i])
            for l in range(depth + 1):
                k = w[2 * l]
                W_np[l][i, : k.shape[0], : k.shape[1]] = k
            for j, u in enumerate(units):
                m_np[j][i, 0, :u] = 1.0
        lr = np.array([g.learning_rate for g in genomes], np.float32).reshape(P, 1, 1)
        idxs = [batch_indices(s, n, st.batch, steps) for s in seeds]
        xb = np.stack([data["x_train"][ix] for ix in idxs], axis=1)                       # (S, P, B, dim)
        yb = np.stack([data["y_train"][ix] for ix in idxs], axis=1)                       # (S, P, B)
        with tfenv.device_scope(tf, device):
            Wv, bv = [tf.Variable(w) for w in W_np], [tf.Variable(b) for b in b_np]
            masks = [tf.constant(m) for m in m_np]
            tvars = Wv + bv
            opt = BatchedAdam(tf, lr, tvars)
            xbt, ybt = tf.constant(xb), tf.constant(yb)
            tfenv.sync(Wv[-1], xbt, ybt)
            comps["construct"] = pc() - t
            fns = [_activation(tf, a) for a in acts]

            def forward(x):
                h = x
                for l in range(depth):
                    h = fns[l](tf.matmul(h, Wv[l]) + bv[l]) * masks[l]
                return (tf.matmul(h, Wv[depth]) + bv[depth])[..., 0]                     # (P, B)

            @tf.function(jit_compile=False)
            def step(x, y):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_sum(tf.reduce_mean(tf.square(y - forward(x)), axis=1))
                opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
                return loss

            t = pc()
            step.get_concrete_function(tf.TensorSpec((P, st.batch, st.in_dim), tf.float32), tf.TensorSpec((P, st.batch), tf.float32))
            comps["trace"] = pc() - t
            t = pc()
            loss = step(xbt[0], ybt[0])
            tfenv.sync(loss, Wv[-1])
            comps["first_exec"] = pc() - t
            t = pc()
            for i in range(1, steps):
                loss = step(xbt[i], ybt[i])
            if steps > 1:
                tfenv.sync(loss, Wv[-1])
                comps["train"] = pc() - t
            t = pc()
            xv = tf.constant(np.broadcast_to(data["x_val"], (P,) + data["x_val"].shape).copy())
            pred = forward(xv).numpy()                                                    # (P, n_val)
            comps["fitness"] = pc() - t
        for i, r in enumerate(results):
            r2 = r2_score(data["y_val"], pred[i])
            if math.isfinite(r2):
                r["val_r2"] = r2
            else:
                r["status"], r["error"] = "failed", "FloatingPointError: non-finite fitness (diverged)"
    except LimitExceeded:
        raise
    except Exception as e:
        if tfenv.is_placement_error(e):
            raise
        for r in results:
            r["status"], r["error"] = "failed", f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        t = pc()
        Wv = bv = masks = tvars = opt = xbt = ybt = step = forward = loss = pred = None
        comps["teardown"] = pc() - t
    share = {k: v / P for k, v in comps.items()}                      # bucket cost is shared equally by its members
    for r in results:
        r["comps"] = dict(share)
    return results


def eval_batched_population(tf, device: str, genomes: List[G.Genome], data, st: Settings, steps: int,
                            check: Callable[[], None]) -> List[Dict[str, Any]]:
    """Bucket by (depth, activations); buckets of >= 2 are batched, singletons run sequentially (coverage is reported)."""
    out: List[Optional[Dict[str, Any]]] = [None] * len(genomes)
    for idxs in bucket_genomes(genomes):
        check()
        if len(idxs) >= 2:
            for i, r in zip(idxs, eval_bucket_batched(tf, device, [genomes[i] for i in idxs], data, st, steps)):
                out[i] = r
        else:
            out[idxs[0]] = eval_individual(tf, device, genomes[idxs[0]], data, st, steps)
    return out                                                       # type: ignore[return-value]


# -------------------------------------------------------------------------------------------- cells
class Ctx:
    """Execution context shared by all cells of a block."""

    def __init__(self, settings: Settings, device: str, monitor=None, tf=None, evaluator=None, placement: str = "strict"):
        self.settings, self.device, self.monitor, self.tf = settings, device, monitor, tf
        self.evaluator, self.placement = evaluator, placement
        self.deadline = float("inf")
        self._data: Optional[Dict[str, np.ndarray]] = None

    @property
    def data(self):
        if self._data is None:
            self._data = dataset_for(self.settings)
        return self._data

    def check(self) -> None:
        if self.monitor is not None and self.monitor.aborted:
            raise LimitExceeded(self.monitor.reason)
        if time.monotonic() > self.deadline:
            raise CellTimeout()


def make_eval_batch(ctx: Ctx, cell: Cell) -> Callable[[List[G.Genome]], List[Dict[str, Any]]]:
    st, steps, mode = ctx.settings, cell.point.steps, cell.strategy.mode

    if mode == "seq":
        def run(genomes):
            out = []
            for g in genomes:
                ctx.check()
                out.append(eval_individual(ctx.tf, ctx.device, g, ctx.data, st, steps))
            return out
    elif mode == "batched":
        def run(genomes):
            return eval_batched_population(ctx.tf, ctx.device, genomes, ctx.data, st, steps, ctx.check)
    else:
        def run(genomes):
            return ctx.evaluator.evaluate(genomes, steps, ctx.check)
    return run


def run_cell(ctx: Ctx, cell: Cell, cell_limit_s: float) -> Dict[str, Any]:
    """Run one configuration (a short evolution) and return a fully decomposed result record.  Never raises for a
    cell-level time-out: that is recorded as ``aborted`` (and must not be used as a measurement)."""
    st, pt, strat = ctx.settings, cell.point, cell.strategy
    rec: Dict[str, Any] = {"cell_id": cell.cell_id, "point": pt.as_dict(), "strategy": {"device": strat.device, "mode": strat.mode, "workers": strat.workers},
                           "phase": cell.phase, "rep": cell.rep, "status": "completed", "reason": "", "placement": ctx.placement,
                           "signature": workload_signature(pt, st), "predicted_s": None}
    ctx.deadline = time.monotonic() + cell_limit_s
    space, pop_g = make_space(pt, st), initial_population(pt, st)
    ga_rng = random.Random(st.seed * 31 + pt.population)
    pop = [{"g": g, "score": None, "done": False} for g in pop_g]
    uid_next = len(pop)
    comps_tot = {k: 0.0 for k in COMP_KEYS + ("ga", "cleanup")}
    gens: List[Dict[str, Any]] = []
    evals = failed = 0
    best_batched = {"batched_members": 0, "padded": []}
    mon = ctx.monitor
    if mon is not None:
        mon.window_peaks()
    s0, sw0 = take_sample(), swap_counters()
    extra_start = extra_gb(mon.baseline, s0) if mon is not None else 0.0
    eval_batch = make_eval_batch(ctx, cell)
    t_cell = pc()
    try:
        for gen in range(pt.generations):
            ctx.check()
            t_gen = pc()
            todo = [x for x in pop if not x["done"]]
            results = eval_batch([x["g"] for x in todo])
            for x, r in zip(todo, results):
                x["done"], x["score"] = True, (r["val_r2"] if r["status"] == "ok" else None)
                for k in COMP_KEYS:
                    comps_tot[k] += r["comps"][k]
                failed += r["status"] != "ok"
                if r.get("batched"):
                    best_batched["batched_members"] += 1
                    best_batched["padded"].append(r.get("padded_ratio", 1.0))
            evals += len(todo)
            ga_s = 0.0
            if gen < pt.generations - 1:
                t = pc()
                offspring, uid_next = G.next_generation([x["g"] for x in pop], [x["score"] for x in pop], ga_rng, space,
                                                        elite_k=elite_count(pt.population), tournament_k=3, crossover_rate=0.7,
                                                        next_uid=uid_next, generation=gen + 1)
                pop = [pop[o.primary_parent] if o.is_elite else {"g": o.genome, "score": None, "done": False} for o in offspring]
                ga_s = pc() - t
            comps_tot["ga"] += ga_s
            t = pc()
            gc.collect()
            cl = pc() - t
            comps_tot["cleanup"] += cl
            scored = [r["val_r2"] for r in results if r["val_r2"] is not None]
            gens.append({"gen": gen, "evaluated": len(todo), "wall_s": pc() - t_gen, "ga_s": ga_s, "cleanup_s": cl,
                         "best_val_r2": max(scored) if scored else None})
    except CellTimeout:
        rec.update(status="aborted", reason=f"per-configuration time limit of {cell_limit_s:g}s reached")
    except LimitExceeded as e:
        rec.update(status="aborted", reason=e.reason)
        rec["completed_generations"] = len(gens)
        rec["_limit"] = True                                 # the block runner re-raises after storing this record
    wall = sum(g["wall_s"] for g in gens) if rec["status"] == "completed" else pc() - t_cell
    s1, sw1 = take_sample(), swap_counters()
    peaks = mon.window_peaks() if mon is not None else {}
    train_wall = comps_tot["train"] + comps_tot["first_exec"]
    attributed = sum(comps_tot.values()) if strat.mode != "mp" else None
    scores = [g["best_val_r2"] for g in gens if g["best_val_r2"] is not None]
    rec.update({
        "evaluations": evals, "failed_evaluations": failed, "wall_s": wall, "generations_run": len(gens), "gens": gens,
        "comps": {**comps_tot, "unattributed": (wall - attributed) if attributed is not None else None},
        "comps_basis": "worker-seconds (summed over workers)" if strat.mode == "mp" else "wall-seconds",
        "ips": (evals / wall) if (wall > 0 and rec["status"] == "completed") else None,
        "steps_per_s": (evals * pt.steps / wall) if (wall > 0 and rec["status"] == "completed") else None,
        "train_steps_per_s": (evals * (pt.steps - 1) / comps_tot["train"]) if comps_tot["train"] > 0 and pt.steps > 1 else None,
        "best_val_r2": max(scores) if scores else None,
        "peak_extra_gb_abs": peaks.get("max_extra_gb"), "peak_additional_gb": (peaks["max_extra_gb"] - extra_start) if peaks else None,
        "min_avail_gb": peaks.get("min_avail_gb"), "max_sys_pct": peaks.get("max_sys_pct"),
        "peak_rss_gb": peaks.get("max_rss_gb"), "swap_growth_mb": (sw1["used_gb"] - sw0["used_gb"]) * 1024,
        "rss_start_gb": s0.rss_gb, "rss_end_gb": s1.rss_gb, "retained_gb": s1.rss_gb - s0.rss_gb,
        "batched_member_fraction": (best_batched["batched_members"] / evals) if (evals and strat.mode == "batched") else None,
        "padded_ratio_mean": float(np.mean(best_batched["padded"])) if best_batched["padded"] else None,
        "train_fraction": (train_wall / wall) if wall else None,
    })
    if strat.mode == "mp" and ctx.evaluator is not None:
        rec["mp"] = {"workers": strat.workers, "startup_s": ctx.evaluator.startup_s, "threads_per_worker": ctx.evaluator.threads,
                     "peak_children_rss_gb": ctx.evaluator.peak_children_rss_gb,
                     "parallel_efficiency": (sum(comps_tot[k] for k in COMP_KEYS) / (wall * strat.workers)) if wall else None}
    return rec


# ------------------------------------------------------------------------------ validation + block runner
def verify_device(tf, device: str, st: Settings, rec) -> Optional[str]:
    """Placement evidence + float64 forward-pass gate + batched-vs-sequential equivalence.  Returns an error or None."""
    from types import SimpleNamespace
    from ..evolution import evaluate as evo_evaluate, make_dataset
    from ..workloads import numeric_gate
    probe = G.Genome([G.LayerGene(8, "relu")], 1e-3)
    with tfenv.device_scope(tf, device):
        m = build_keras(tf, probe, st.in_dim, 0)
    ev = tfenv.placement_evidence(tf, device, m.trainable_variables, {})
    rec.data["device_evidence"] = ev
    if not ev["verified"]:
        return "device placement could not be verified: " + "; ".join(ev["problems"])
    data = make_dataset(512, 256, 64, st.in_dim, st.seed)
    ref = G.Genome([G.LayerGene(32, "relu"), G.LayerGene(32, "tanh")], 1e-3, uid=-1)
    r = evo_evaluate(tf, device, ref, data, SimpleNamespace(seed=st.seed, ne_epochs=1, ne_batch=st.batch, optimizer="simple"), trace_steps=1)
    rec.data["forward_diagnosis"] = r.diag
    return numeric_gate(r.diag)


def selfcheck_batched(tf, device: str, st: Settings, rec) -> Optional[str]:
    """Batched (padded, masked) training must agree with sequential training on THIS device and graph path."""
    data = dataset_for(st)
    gs = [G.Genome([G.LayerGene(32, "relu"), G.LayerGene(64, "tanh")], 2e-3, uid=901),
          G.Genome([G.LayerGene(64, "relu"), G.LayerGene(32, "tanh")], 1e-3, uid=902),
          G.Genome([G.LayerGene(48, "relu"), G.LayerGene(48, "tanh")], 3e-3, uid=903)]
    seq = [eval_individual(tf, device, g, data, st, 4)["val_r2"] for g in gs]
    bat = [r["val_r2"] for r in eval_bucket_batched(tf, device, gs, data, st, 4)]
    diffs = [abs(a - b) for a, b in zip(seq, bat) if a is not None and b is not None]
    rec.data["batched_selfcheck"] = {"sequential": seq, "batched": bat, "max_abs_diff": max(diffs) if diffs else None}
    if not diffs or max(diffs) > 5e-3:
        return f"batched training disagrees with sequential training on {device} (max |dR2| = {max(diffs) if diffs else 'n/a'})"
    return None


def record_cell(rec, cell_rec: Dict[str, Any]) -> None:
    rec.data.setdefault("cells", []).append(cell_rec)
    p, s = cell_rec["point"], cell_rec["strategy"]
    ctx = dict(device=s["device"], workload=f'{p["tier"]}/P{p["population"]}/S{p["steps"]}/{p["mode"]}',
               policy=cell_rec["strategy"]["mode"] if s["mode"] != "mp" else f'mp{s["workers"]}', repeat=cell_rec["rep"],
               phase=cell_rec["phase"], tier=p["tier"], population=p["population"], steps=p["steps"], placement=cell_rec["placement"])
    if cell_rec["status"] != "completed":
        rec.measure("cell_status", cell_rec["status"], "", **ctx)
        return
    for name, key, unit, kind in (("individuals_per_s", "ips", "ind/s", "calculated"), ("wall_s", "wall_s", "s", "measured"),
                                  ("steps_per_s", "steps_per_s", "steps/s", "calculated"),
                                  ("peak_additional_gb", "peak_additional_gb", "GB", "measured"),
                                  ("swap_growth_mb", "swap_growth_mb", "MB", "measured"),
                                  ("retained_gb", "retained_gb", "GB", "measured")):
        if cell_rec.get(key) is not None:
            rec.measure(name, cell_rec[key], unit, kind=kind, **ctx)
    for k, v in cell_rec["comps"].items():
        if v is not None:
            rec.measure(f"component_{k}_s", v, "s", **ctx)


def run_block(args, rec, monitor, job, placement: str, plan_path: str) -> None:
    """Worker entry for ``scalingblock@<device>:<seq|batched|mpN>``: runs the planned cells in this one process."""
    with open(plan_path) as fh:
        plan = json.load(fh)
    st = Settings.from_dict(plan["settings"])
    cells = [Cell.from_dict(c) for c in plan["cells"]]
    device, variant = job.device, job.variant or "seq"
    ctx = Ctx(st, device, monitor, placement=placement)
    rec.data["block"] = {"device": device, "variant": variant, "n_cells": len(cells), "placement": placement,
                         "cell_limit_s": plan.get("cell_limit_s", st.max_cell_s)}
    cell_limit = plan.get("cell_limit_s", st.max_cell_s)
    t_cold = pc()
    if variant.startswith("mp"):
        from .pool import MPEvaluator, max_workers_by_memory
        w = int(variant[2:])
        cap = max_workers_by_memory(args.max_gb, requested=w)
        if cap < w:
            raise RuntimeError(f"{w} workers exceed the memory-derived cap of {cap} for --max-gb {args.max_gb}")
        ctx.evaluator = MPEvaluator(w, st, rss_cap_total_gb=args.max_gb * 0.8)
        rec.data["block"]["pool_startup_s"] = ctx.evaluator.startup_s
    else:
        ctx.tf = tfenv.configure(device, placement, st.grappler)
        rec.data["tf"] = tfenv.tf_info(ctx.tf)
        bad = verify_device(ctx.tf, device, st, rec)
        if bad is None and variant == "batched":
            bad = selfcheck_batched(ctx.tf, device, st, rec)
        if bad:
            rec.set_status("failed", bad)
            rec.note("Numerical/placement validation failed: no timings are reported from this device/strategy.")
            return
        warm = Cell(Point("small", 2, 2, "homo" if variant == "batched" else "hetero", 1), Strategy(device, variant), "warmup")
        run_cell(ctx, warm, cell_limit)                       # discarded: absorbs runtime / shader-cache cold start
    rec.data["block"]["cold_start_s"] = pc() - t_cold
    aborted_floor: Dict[Tuple[str, str], float] = {}
    try:
        for cell in cells:
            key = (cell.strategy.name, cell.point.tier)
            cost = cell.point.population * cell.point.steps * cell.point.generations
            if key in aborted_floor and cost >= aborted_floor[key]:
                record_cell(rec, {"cell_id": cell.cell_id, "point": cell.point.as_dict(), "strategy": {"device": device, "mode": cell.strategy.mode, "workers": cell.strategy.workers},
                                  "phase": cell.phase, "rep": cell.rep, "status": "skipped", "placement": placement,
                                  "reason": "an equal or cheaper configuration of this strategy/tier hit the time limit"})
                continue
            if monitor is not None and monitor.aborted:
                raise LimitExceeded(monitor.reason)
            print(f"  cell {cell.cell_id}", flush=True)
            try:
                cr = run_cell(ctx, cell, cell_limit)
            except Exception as e:                                           # noqa: BLE001
                if tfenv.is_placement_error(e) or isinstance(e, LimitExceeded):
                    raise
                cr = {"cell_id": cell.cell_id, "point": cell.point.as_dict(), "strategy": {"device": device, "mode": cell.strategy.mode, "workers": cell.strategy.workers},
                      "phase": cell.phase, "rep": cell.rep, "status": "failed", "placement": placement,
                      "reason": f"{type(e).__name__}: {str(e)[:200]}"}
            limit_hit = cr.pop("_limit", False)
            record_cell(rec, cr)
            rec.checkpoint()
            if limit_hit:
                raise LimitExceeded(cr["reason"])
            if cr["status"] == "aborted":
                aborted_floor[key] = cost
                if variant.startswith("mp"):
                    break                                                    # the pool was terminated by the abort
    finally:
        if ctx.evaluator is not None:
            ctx.evaluator.close()
