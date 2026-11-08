"""Command-line interface: argument parsing, --dry-run, supervisor and worker."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psutil

from . import __version__, genetics
from .common import Recorder, atomic_write_json, sanitize
from .limits import (GB, LimitExceeded, ResourceLimits, ResourceMonitor, capture_baseline, check_limits,
                     memory_headroom_gb, preflight, take_sample)
from .workloads import perf_workload

LEGACY = ("alloc", "train", "leak")
NEW = ("performance", "neuroevolution", "memdiag", "scaling")
ALL_TESTS = LEGACY + NEW
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tf_memory_stress_test.py")

QUICK = dict(perf_steps=30, perf_warmup=5, perf_repeats=3, perf_width=1024, perf_depth=3, perf_batch=128,
             perf_pool=4, perf_workloads=["mlp"], ne_pop=6, ne_generations=3, ne_elite=1, ne_epochs=3,
             ne_train_n=1024, ne_val_n=256, ne_test_n=256, ne_max_units=128, ne_max_layers=3,
             memdiag_cycles=10, leak_epochs=10,
             scaling_pops=[4, 8], scaling_pilot_pop=4, scaling_pilot_steps=[1, 20], scaling_generations=2, scaling_confirm_points=1,
             scaling_workers=[2], scaling_max_total_min=10.0, scaling_max_cell_min=2.5)


# ---------------------------------------------------------------------------- jobs
@dataclass(frozen=True)
class Job:
    test: str
    device: Optional[str] = None
    variant: Optional[str] = None      # workload (performance) or cleanup policy (memdiag)

    @property
    def name(self) -> str:
        n = self.test
        if self.device:
            n += f"@{self.device}"
        if self.variant:
            n += f":{self.variant}"
        return n

    @staticmethod
    def parse(name: str) -> "Job":
        variant = device = None
        if ":" in name:
            name, variant = name.split(":", 1)
        if "@" in name:
            name, device = name.split("@", 1)
        return Job(name, device, variant)


def plan_jobs(args) -> List[Tuple[str, List[Job]]]:
    plan: List[Tuple[str, List[Job]]] = []
    for t in args.tests:
        if t in LEGACY:
            plan.append((t, [Job(t)]))
        elif t == "performance":
            # interleave devices per workload so slow thermal drift does not favour one device
            plan.append((t, [Job(t, d, w) for w in args.perf_workloads for d in args.perf_devices]))
        elif t == "neuroevolution":
            plan.append((t, [Job(t, d) for d in args.ne_devices]))
        elif t == "memdiag":
            plan.append((t, [Job(t, d, p) for d in args.memdiag_devices for p in args.memdiag_policies]))
        elif t == "scaling":
            plan.append((t, [Job("scaling")]))                       # adaptive: the controller starts the actual worker blocks
    return plan


# ------------------------------------------------------------------------- parsing
def build_parser(total_gb: float) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tf_memory_stress_test.py", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="TensorFlow memory-stress, performance and neuroevolution benchmark (macOS / Apple Silicon).",
        epilog="""examples:
  python tf_memory_stress_test.py --tests performance neuroevolution --quick --dry-run
  python tf_memory_stress_test.py --tests performance --quick
  python tf_memory_stress_test.py --tests performance neuroevolution --max-gb 6
  python tf_memory_stress_test.py --tests memdiag --quick""")
    g = p.add_argument_group("what to run")
    g.add_argument("--tests", nargs="+", choices=ALL_TESTS, default=list(LEGACY),
                   help="legacy: alloc train leak | new: performance neuroevolution memdiag (default: legacy)")
    g.add_argument("--quick", action="store_true", help="conservative small workloads (explicit flags still win)")
    g.add_argument("--dry-run", action="store_true", help="validate configuration, estimate workload, train nothing")
    g.add_argument("--out-dir", default=None, help="report directory (default: bench_results/<UTC timestamp>)")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--optimizer", choices=["simple", "keras"], default="simple",
                   help="simple: float32-only Adam with Keras' update rule (runs under strict GPU placement). "
                        "keras: Keras legacy Adam (its int64 step counter forces soft placement on tensorflow-metal)")
    g.add_argument("--grappler", choices=["default", "no-remapper", "none"], default="default",
                   help="TensorFlow graph-optimizer setting applied to ALL jobs (CPU and GPU alike). no-remapper "
                        "disables MatMul+BiasAdd+activation fusion, none disables graph rewriting; use if the numerics "
                        "check shows wrong graph-mode results on the GPU")
    g.add_argument("--skip-numeric-gate", action="store_true",
                   help="do NOT refuse to report jobs whose forward pass disagrees with float64 truth (not recommended)")
    g.add_argument("--placement", choices=["auto", "strict", "soft"], default="auto",
                   help="GPU op placement. strict: unsupported GPU ops raise. soft: TF may run them on CPU. "
                        "auto: strict first; if strict fails on an unsupported op, retry once in soft mode and "
                        "label every result 'soft' (never silent)")

    s = p.add_argument_group("safety limits (never raised automatically)")
    s.add_argument("--max-gb", type=float, default=None, help="max EXTRA system memory (default: min(6, 50%% RAM))")
    s.add_argument("--min-avail-gb", type=float, default=2.0, help="stop if available memory falls below this")
    s.add_argument("--max-sys-pct", type=float, default=80.0, help="stop if system memory use reaches this %%")
    s.add_argument("--max-swap-mb", type=float, default=512.0, help="stop if swap grows by this much")
    s.add_argument("--max-swap-gb", type=float, default=None, help=argparse.SUPPRESS)  # deprecated alias
    s.add_argument("--max-runtime-min", type=float, default=20.0, help="runtime budget per benchmark")
    s.add_argument("--max-generations", type=int, default=10, help="hard cap for --ne-generations")

    f = p.add_argument_group("performance")
    f.add_argument("--perf-workloads", nargs="+", choices=["mlp", "convnet"], default=["mlp", "convnet"])
    f.add_argument("--perf-devices", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    f.add_argument("--perf-mode", choices=["graph", "eager"], default="graph")
    f.add_argument("--perf-batch", type=int, default=256)
    f.add_argument("--perf-steps", type=int, default=100, help="timed steps per repeat")
    f.add_argument("--perf-warmup", type=int, default=10)
    f.add_argument("--perf-repeats", type=int, default=5)
    f.add_argument("--perf-pool", type=int, default=8, help="distinct synthetic batches cycled through")
    f.add_argument("--perf-width", type=int, default=2048)
    f.add_argument("--perf-depth", type=int, default=4)
    f.add_argument("--perf-in-dim", type=int, default=784)
    f.add_argument("--perf-classes", type=int, default=10)

    n = p.add_argument_group("neuroevolution")
    n.add_argument("--ne-devices", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    n.add_argument("--ne-pop", type=int, default=12)
    n.add_argument("--ne-generations", type=int, default=5)
    n.add_argument("--ne-elite", type=int, default=2)
    n.add_argument("--ne-tournament", type=int, default=3)
    n.add_argument("--ne-crossover", type=float, default=0.7)
    n.add_argument("--ne-epochs", type=int, default=5, help="fixed budget: passes over the training set")
    n.add_argument("--ne-batch", type=int, default=128)
    n.add_argument("--ne-train-n", type=int, default=4096)
    n.add_argument("--ne-val-n", type=int, default=1024)
    n.add_argument("--ne-test-n", type=int, default=1024)
    n.add_argument("--ne-in-dim", type=int, default=32)
    n.add_argument("--ne-max-layers", type=int, default=4)
    n.add_argument("--ne-activations", nargs="+", default=["relu", "tanh", "sigmoid", "swish"],
                   choices=["relu", "tanh", "sigmoid", "swish", "elu", "gelu", "selu", "softplus"],
                   help="activation search space (default excludes elu: tensorflow-metal has no GPU kernel for it, "
                        "which would force soft placement; the same space is used on CPU and GPU)")
    n.add_argument("--ne-max-units", type=int, default=256, choices=[8, 16, 32, 64, 128, 256, 512])
    n.add_argument("--ne-workers", type=int, default=1, help="parallel evaluation threads (1 = sequential)")
    n.add_argument("--ne-inherit-weights", action="store_true")
    n.add_argument("--ne-cleanup", choices=["each", "generation", "gc", "never"], default="gc",
                   help="each/generation: clear_session()+gc.collect() per individual / per generation; gc: gc.collect() "
                        "only per generation; never: no cleanup. On tensorflow-metal clear_session() grew system memory by "
                        "~100 MB per call in the author's memdiag run")
    n.add_argument("--memdiag-devices", nargs="+", choices=["cpu", "gpu"], default=["gpu"])
    n.add_argument("--memdiag-policies", nargs="+", choices=["none", "gc", "clear"], default=["none", "gc"],
                   help="cleanup policies to compare (each in its own process). 'clear' is opt-in: on tensorflow-metal it "
                        "reached the 80%% system-memory cap in ~17 cycles on the author's M1 Pro")
    n.add_argument("--memdiag-cycles", type=int, default=15)
    n.add_argument("--memdiag-threshold-mb", type=float, default=1.0)

    sc = p.add_argument_group("scaling (CPU/GPU crossover for neuroevolution; adaptive design, never the full Cartesian product)")
    sc.add_argument("--scaling-tiers", nargs="+", choices=["small", "medium", "large"], default=["small", "medium", "large"],
                    help="small: 1-2 layers x 32-64 | medium: 2-4 x 128-256 | large: 4-6 x 512-1024 hidden units")
    sc.add_argument("--scaling-pops", nargs="+", type=int, default=[4, 8, 16, 32], help="population sizes explored during expansion")
    sc.add_argument("--scaling-steps", nargs="+", type=int, default=[1, 5, 20, 50], help="training steps per individual")
    sc.add_argument("--scaling-pilot-steps", nargs="+", type=int, default=None, help="steps used in the pilot (default: all --scaling-steps)")
    sc.add_argument("--scaling-pilot-pop", type=int, default=8)
    sc.add_argument("--scaling-generations", type=int, default=3)
    sc.add_argument("--scaling-confirm-reps", type=int, default=3, help="independent repetitions (fresh processes) per confirmed point")
    sc.add_argument("--scaling-confirm-points", type=int, default=3, help="max crossover candidates to confirm")
    sc.add_argument("--scaling-batch", type=int, default=128)
    sc.add_argument("--scaling-in-dim", type=int, default=32)
    sc.add_argument("--scaling-train-n", type=int, default=4096)
    sc.add_argument("--scaling-val-n", type=int, default=1024)
    sc.add_argument("--scaling-activations", nargs="+", default=["relu", "tanh"], choices=["relu", "tanh", "sigmoid", "swish"],
                    help="few activations => more batch-compatible buckets; elu is excluded (no Metal kernel)")
    sc.add_argument("--scaling-workers", nargs="+", type=int, default=[2, 4], help="CPU multiprocessing worker counts to test")
    sc.add_argument("--scaling-max-workers", type=int, default=4, help="hard cap on CPU worker processes")
    sc.add_argument("--scaling-max-cell-min", type=float, default=5.0, help="time limit per configuration")
    sc.add_argument("--scaling-max-total-min", type=float, default=30.0, help="time limit for the whole scaling benchmark")
    sc.add_argument("--scaling-extend-steps", nargs="+", type=int, default=[], help="OPT-IN larger step counts tried at the best candidate")
    sc.add_argument("--scaling-skip-batched", action="store_true")
    sc.add_argument("--scaling-skip-mp", action="store_true")
    sc.add_argument("--scaling-allow-gpu-multiprocess", action="store_true",
                    help="NOT SUPPORTED: GPU multiprocessing would make processes compete for the Metal device / unified memory")

    lg = p.add_argument_group("legacy tests (alloc / train / leak)")
    lg.add_argument("--cpu", action="store_true", help="legacy tests: hide the GPU")
    lg.add_argument("--step-mb", type=int, default=256)
    lg.add_argument("--start-batch", type=int, default=256)
    lg.add_argument("--max-batch", type=int, default=65536)
    lg.add_argument("--slowdown", type=float, default=2.5)
    lg.add_argument("--steps", type=int, default=20)
    lg.add_argument("--width", type=int, default=2048)
    lg.add_argument("--depth", type=int, default=6)
    lg.add_argument("--in-dim", type=int, default=1024)
    lg.add_argument("--classes", type=int, default=100)
    lg.add_argument("--leak-epochs", type=int, default=20)
    lg.add_argument("--leak-mb-per-epoch", type=float, default=5.0)

    h = p.add_argument_group("internal")
    h.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    h.add_argument("--run-placement", default="strict", help=argparse.SUPPRESS)
    h.add_argument("--result-path", default=None, help=argparse.SUPPRESS)
    h.add_argument("--budget-s", type=float, default=None, help=argparse.SUPPRESS)
    h.add_argument("--plan-path", default=None, help=argparse.SUPPRESS)
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    total_gb = psutil.virtual_memory().total / GB
    p = build_parser(total_gb)
    if "--quick" in argv:
        p.set_defaults(**QUICK)
    args = p.parse_args(argv)
    if args.max_gb is None:
        args.max_gb = min(6.0, total_gb * 0.5)
    if args.max_swap_gb is not None:
        args.max_swap_mb = args.max_swap_gb * 1024
    if args.out_dir is None:
        args.out_dir = os.path.join("bench_results", _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    return args


def scaling_settings(args):
    from .scaling.pool import max_workers_by_memory
    from .scaling.workload import Settings
    return Settings(
        seed=args.seed, batch=args.scaling_batch, in_dim=args.scaling_in_dim, train_n=args.scaling_train_n, val_n=args.scaling_val_n,
        generations=args.scaling_generations, activations=tuple(args.scaling_activations), tiers=tuple(args.scaling_tiers),
        pops=tuple(sorted(set(args.scaling_pops))), steps=tuple(sorted(set(args.scaling_steps))),
        pilot_pop=args.scaling_pilot_pop, pilot_steps=tuple(sorted(set(args.scaling_pilot_steps or args.scaling_steps))),
        confirm_reps=args.scaling_confirm_reps, confirm_points=args.scaling_confirm_points, max_cell_s=args.scaling_max_cell_min * 60.0,
        max_total_s=args.scaling_max_total_min * 60.0, workers=tuple(args.scaling_workers),
        max_workers=max_workers_by_memory(args.max_gb, requested=args.scaling_max_workers),
        extend_steps=tuple(args.scaling_extend_steps), skip_batched=args.scaling_skip_batched, skip_mp=args.scaling_skip_mp,
        grappler=args.grappler, quick=args.quick)


def scaling_preflight(args, limits: ResourceLimits, sample) -> List[str]:
    """Refuse a scaling run that cannot make a single paired CPU/GPU comparison with the memory available RIGHT NOW."""
    from .scaling import design as sd
    st = scaling_settings(args)
    head = memory_headroom_gb(limits, sample)
    need = {}
    for c in sd.pilot_cells(st):
        need[c.strategy.device] = min(need.get(c.strategy.device, 1e9), sd.estimate_memory_gb(c.point, c.strategy, st) * 1.5)
    if head >= max(need.values()):
        return []
    worst = max(need, key=need.get)
    return [f"memory headroom is only {head:.1f} GB (system use {sample.sys_pct:.0f}%, {sample.avail_gb:.1f} GB available), but even the smallest "
            f"{worst} configuration needs ~{need[worst]:.1f} GB (estimate x1.5). No paired CPU/GPU comparison would be possible. "
            "Close other applications (browsers, IDEs, VMs) and retry; `--dry-run` shows the current headroom. Limits are never raised automatically."]


def limits_from(args) -> ResourceLimits:
    return ResourceLimits(max_extra_gb=args.max_gb, min_avail_gb=args.min_avail_gb, max_sys_pct=args.max_sys_pct,
                          max_swap_mb=args.max_swap_mb, max_runtime_s=args.max_runtime_min * 60.0)


def validate_args(args) -> List[str]:
    """Configuration errors.  Limits are NEVER silently raised to make a config fit."""
    errs = limits_from(args).problems()
    if args.ne_generations > args.max_generations:
        errs.append(f"--ne-generations {args.ne_generations} exceeds --max-generations {args.max_generations} "
                    "(raise --max-generations explicitly if you really want this)")
    for name in ("ne_generations", "ne_pop", "ne_epochs", "ne_batch", "perf_steps", "perf_repeats",
                 "perf_batch", "perf_pool", "ne_train_n", "ne_val_n", "ne_test_n", "memdiag_cycles"):
        if getattr(args, name) < 1:
            errs.append(f"--{name.replace('_', '-')} must be >= 1")
    if args.perf_warmup < 0:
        errs.append("--perf-warmup must be >= 0")
    if "performance" in args.tests and args.perf_repeats < 3:
        errs.append("--perf-repeats must be >= 3 for meaningful variability statistics")
    if "neuroevolution" in args.tests:
        if args.ne_pop < 2:
            errs.append("--ne-pop must be >= 2")
        if not 0 <= args.ne_elite < args.ne_pop:
            errs.append("--ne-elite must satisfy 0 <= elite < population")
        if not 1 <= args.ne_workers <= 4:
            errs.append("--ne-workers must be in [1, 4]")
        if not 0 <= args.ne_crossover <= 1:
            errs.append("--ne-crossover must be in [0, 1]")
    if "scaling" in args.tests:
        if args.scaling_allow_gpu_multiprocess:
            errs.append("--scaling-allow-gpu-multiprocess is not supported: several processes would compete for the Metal GPU and "
                        "unified memory. GPU strategies run in one process (sequential or batched).")
        if args.scaling_confirm_reps < 2:
            errs.append("--scaling-confirm-reps must be >= 2 (a confidence interval needs at least two independent repetitions)")
        if args.scaling_pilot_pop < 2 or any(v < 1 for v in args.scaling_steps + args.scaling_pops):
            errs.append("--scaling-pilot-pop must be >= 2 and --scaling-steps/--scaling-pops must be >= 1")
        if args.scaling_max_workers < 1 or any(w < 1 for w in args.scaling_workers):
            errs.append("worker counts must be >= 1")
        if args.scaling_max_cell_min * 60 > args.scaling_max_total_min * 60:
            errs.append("--scaling-max-cell-min cannot exceed --scaling-max-total-min")
        if args.scaling_generations < 2:
            errs.append("--scaling-generations must be >= 2 so mutation / crossover / selection are part of the measured work")
        if args.scaling_generations > args.max_generations:
            errs.append(f"--scaling-generations {args.scaling_generations} exceeds --max-generations {args.max_generations}")
        if args.scaling_pilot_pop not in args.scaling_pops and args.scaling_pilot_pop > max(args.scaling_pops):
            errs.append("--scaling-pilot-pop is larger than every --scaling-pops value")
    if "memdiag" in args.tests and args.memdiag_cycles < 9:
        errs.append("--memdiag-cycles must be >= 9 (3 warm-up + >= 6 analysed)")
    return errs


# -------------------------------------------------------------------------- dry run
def estimate_plan(args) -> Dict[str, Any]:
    plan: Dict[str, Any] = {}
    if "performance" in args.tests:
        rows = []
        for w in args.perf_workloads:
            wl = perf_workload(w, args.perf_in_dim, args.perf_width, args.perf_depth, args.perf_classes)
            est = wl.estimate(args.perf_batch, args.perf_pool)
            steps = 1 + args.perf_warmup + args.perf_repeats * args.perf_steps
            rows.append({"workload": w, **est, "steps_per_job": steps,
                         "samples_per_job": steps * args.perf_batch,
                         "jobs": len(args.perf_devices), "kind": "estimated"})
        plan["performance"] = rows
    if "neuroevolution" in args.tests:
        evals = args.ne_pop + (args.ne_pop - args.ne_elite) * (args.ne_generations - 1)
        steps_per = args.ne_epochs * max(1, args.ne_train_n // args.ne_batch)
        units = max(u for u in (8, 16, 32, 64, 128, 256, 512) if u <= args.ne_max_units)
        worst = genetics.count_params(
            genetics.Genome([genetics.LayerGene(units, "relu")] * args.ne_max_layers, 1e-3), args.ne_in_dim, 1)
        data_gb = (args.ne_train_n * args.ne_epochs + args.ne_val_n + args.ne_test_n) * (args.ne_in_dim + 1) * 4 / GB
        plan["neuroevolution"] = {
            "evaluations_per_device": evals, "train_steps_per_evaluation": steps_per,
            "train_steps_per_device_upper_bound": evals * steps_per, "devices": list(args.ne_devices),
            "worst_case_params": worst, "worst_case_model_and_optimizer_gb": worst * 16 / GB,
            "dataset_gb": data_gb, "workers": args.ne_workers, "kind": "estimated"}
    if "scaling" in args.tests:
        from .scaling.design import design_summary
        plan["scaling"] = design_summary(scaling_settings(args))
    if "memdiag" in args.tests:
        plan["memdiag"] = {"jobs": len(args.memdiag_policies) * len(args.memdiag_devices), "cycles_per_job": args.memdiag_cycles,
                           "kind": "estimated"}
    return plan


def dry_run(args, limits: ResourceLimits) -> int:
    errs = validate_args(args)
    s = take_sample()
    pre = preflight(limits, s)
    if "scaling" in args.tests:
        pre += scaling_preflight(args, limits, s)
    plan = estimate_plan(args)
    print("DRY RUN - nothing will be trained.\n")
    print(f"Tests: {', '.join(args.tests)}   (quick={args.quick}, seed={args.seed}, placement={args.placement})")
    print(f"Limits: extra memory <= {limits.max_extra_gb:.1f} GB | available >= {limits.min_avail_gb:.1f} GB | "
          f"system use < {limits.max_sys_pct:.0f}% | swap growth < {limits.max_swap_mb:.0f} MB | "
          f"runtime <= {limits.max_runtime_s / 60:.0f} min per benchmark")
    print(f"Current system: {s.avail_gb:.1f} GB available | {s.sys_pct:.0f}% used | swap {s.swap_used_gb:.2f} GB used\n")
    for jobs in plan_jobs(args):
        print(f"  {jobs[0]:<15} -> {len(jobs[1])} isolated process(es): {', '.join(j.name for j in jobs[1])}")
    print()
    worst_gb = 0.0
    for r in plan.get("performance", []):
        print(f"  performance/{r['workload']}: {r['params']:,} params | ~{r['train_flops_per_step'] / 1e9:.1f} GFLOP/step "
              f"(estimated) | {r['steps_per_job']} steps ({r['samples_per_job']:,} samples) per job | "
              f"est. memory {r['total_gb']:.2f} GB (model+optimizer {r['model_and_optimizer_gb']:.2f}, "
              f"activations {r['activations_gb']:.2f}, data pool {r['data_pool_gb']:.2f})")
        worst_gb = max(worst_gb, r["total_gb"])
    ne = plan.get("neuroevolution")
    if ne:
        print(f"  neuroevolution: {ne['evaluations_per_device']} evaluations per device x "
              f"{ne['train_steps_per_evaluation']} train steps (<= {ne['train_steps_per_device_upper_bound']:,} steps) | "
              f"worst-case model {ne['worst_case_params']:,} params ~{ne['worst_case_model_and_optimizer_gb'] * 1024:.0f} MB "
              f"(estimated) | dataset ~{ne['dataset_gb'] * 1024:.0f} MB | workers={ne['workers']}")
        worst_gb = max(worst_gb, ne["worst_case_model_and_optimizer_gb"] * max(1, ne["workers"]) + ne["dataset_gb"])
    md = plan.get("memdiag")
    if md:
        print(f"  memdiag: {md['jobs']} processes x {md['cycles_per_job']} build/train/discard cycles")
    sc = plan.get("scaling")
    if sc:
        print(f"  scaling (adaptive; NOT the full grid): exhaustive grid would be {sc['full_cartesian_points']} workload points "
              f"({sc['full_cartesian_cells_seq_only']} sequential cell-runs before strategies/repetitions)")
        print(f"    planned: pilot {sc['pilot_cells']} cells, expansion <= {sc['expansion_cells_max']}, confirmation <= {sc['confirmation_cells']} "
              f"=> at most {sc['total_cell_runs_max']} cell-runs in <= {sc['worker_processes_max']} worker processes")
        lo, hi = sc["estimated_total_s_range"]
        print(f"    estimated runtime ~{sc['estimated_total_s'] / 60:.0f} min (plausible range {lo / 60:.0f}-{hi / 60:.0f} min; PRIORS from the author's M1 Pro, "
              f"refined online) | hard caps: {sc['per_cell_limit_s'] / 60:.0f} min per configuration, {sc['budget_s'] / 60:.0f} min total")
        head = memory_headroom_gb(limits, s)
        print(f"    resource exposure: worst-case single configuration ~{sc['worst_case_cell_memory_gb']:.1f} GB additional memory (estimated); "
              f"current memory headroom {head:.1f} GB (smallest of: extra-memory allowance, available-memory floor, system-use ceiling) - "
              f"configurations whose estimate x1.5 exceeds it are skipped, not run; CPU multiprocessing <= {scaling_settings(args).max_workers} "
              f"spawned workers (1 thread each); GPU runs are single-process; one fresh process per block")
        worst_gb = max(worst_gb, sc["worst_case_cell_memory_gb"] * 1.0)
    if worst_gb > limits.max_extra_gb:
        errs.append(f"estimated peak {worst_gb:.1f} GB exceeds the {limits.max_extra_gb:.1f} GB limit "
                    "(reduce the workload; limits are not raised automatically)")
    print("\nAll figures above are ESTIMATES from closed-form formulas, not measurements; run time is not predicted.")
    for w in pre:
        print(f"WARNING (a real run would refuse to start): {w}")
    for e in errs:
        print(f"ERROR: {e}")
    print("\nConfiguration " + ("INVALID" if errs else "OK"))
    return 2 if errs else 0


# ---------------------------------------------------------------------- supervisor
def _kill(proc: subprocess.Popen, grace: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_job(args, limits: ResourceLimits, out_dir: str, job: Job, placement: str, remaining_s: float,
            extra_args: Sequence[str] = (), tag: str = "") -> Dict[str, Any]:
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    safe = job.name.replace("@", "_").replace(":", "_")
    path = os.path.join(raw_dir, f"{safe}.{placement}{('.' + tag) if tag else ''}.json")
    if os.path.exists(path):
        os.remove(path)
    cmd = [sys.executable, SCRIPT, *sys.argv[1:], "--out-dir", out_dir, "--worker", job.name,
           "--run-placement", placement, "--result-path", path, "--budget-s", f"{remaining_s:.0f}", *extra_args]
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL=os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2"), PYTHONUNBUFFERED="1")
    print(f"\n>>> {job.name} (placement={placement}, budget {remaining_s:.0f}s)", flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env)
    # Backstop limits: the worker enforces the real limits itself; this only fires if the worker is unresponsive
    # (e.g. blocked inside a native Metal call). +1 GB covers the TensorFlow import footprint in its baseline.
    backstop = ResourceLimits(limits.max_extra_gb + 1.0, limits.min_avail_gb, limits.max_sys_pct,
                              limits.max_swap_mb, limits.max_runtime_s)
    base = capture_baseline()
    over_since: Optional[float] = None
    killed_reason = ""
    term_at: Optional[float] = None
    try:
        while True:
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            reason = check_limits(backstop, base, take_sample(), remaining_s + 20)
            over_since = (over_since or now) if reason else None
            if reason and now - over_since > 15 and term_at is None:
                killed_reason, term_at = f"supervisor backstop: {reason}", now
                proc.terminate()
            if term_at is not None and now - term_at > 15:
                proc.kill()
    except KeyboardInterrupt:
        _kill(proc)
        res = _load(path) or {}
        res.update(job=job.name, status="interrupted", reason="Ctrl-C")
        res.setdefault("measurements", [])
        raise _Interrupted(res)
    rc = proc.returncode
    res = _load(path)
    if res is None or res.get("status") == "running":
        res = res or {"job": job.name, "measurements": [], "data": {}, "notes": []}
        res["status"] = "killed" if killed_reason else "crashed"
        res["reason"] = killed_reason or f"worker exited with code {rc} before writing a result (native crash?)"
    elif killed_reason:
        res.setdefault("notes", []).append(killed_reason)
    res.update(test=job.test, device=job.device, workload=job.variant, placement=placement, exit_code=rc,
               supervised_wall_s=time.monotonic() - t0)
    return res


class _Interrupted(Exception):
    def __init__(self, res):
        self.res = res


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def supervise(args, limits: ResourceLimits, out_dir: str) -> Tuple[List[Dict[str, Any]], bool]:
    results: List[Dict[str, Any]] = []
    interrupted = False
    try:
        for test, jobs in plan_jobs(args):
            if test == "scaling":
                job = _run_scaling(args, limits, out_dir)
                results.append(job)
                if job.pop("_reraise", None) is not None:        # Ctrl-C inside the controller: keep the partial report, stop here
                    interrupted = True
                    break
                continue
            t_start = time.monotonic()
            for job in jobs:
                remaining = limits.max_runtime_s - (time.monotonic() - t_start)
                if remaining < 30:
                    results.append({"job": job.name, "test": job.test, "device": job.device, "workload": job.variant,
                                    "status": "skipped", "reason": "runtime budget for this benchmark exhausted",
                                    "measurements": [], "data": {}, "notes": []})
                    continue
                first = "strict" if (job.device == "gpu" and args.placement in ("auto", "strict")) else \
                    ("soft" if args.placement == "soft" else "strict")
                res = run_job(args, limits, out_dir, job, first, remaining)
                if res["status"] == "placement_failed" and args.placement == "auto" and job.device == "gpu":
                    msg = res.get("reason", "")
                    print(f"    strict placement failed ({msg[:160]}...). Retrying ONCE with soft placement; "
                          "results will be labelled placement=soft.", flush=True)
                    remaining = limits.max_runtime_s - (time.monotonic() - t_start)
                    res2 = run_job(args, limits, out_dir, job, "soft", max(remaining, 30))
                    res2["placement_strict_failure"] = msg
                    res2["attempts"] = 2
                    res2.setdefault("notes", []).append(
                        f"{job.name}: strict placement failed on an unsupported op; the result below was produced "
                        "with soft placement, so some ops may have run on the CPU.")
                    res = res2
                results.append(res)
    except _Interrupted as e:
        e.res.update(test=e.res.get("job", "?").split("@")[0].split(":")[0])
        results.append(e.res)
        interrupted = True
    return results, interrupted


def _run_scaling(args, limits: ResourceLimits, out_dir: str) -> Dict[str, Any]:
    """Adaptive scaling benchmark: the controller (TensorFlow-free) decides, one fresh worker process runs each block."""
    from . import sysinfo
    from .scaling import run as scaling_run
    st = scaling_settings(args)
    cell_cap = st.max_cell_s

    def runner(block, remaining_s: float) -> Dict[str, Any]:
        job = Job("scalingblock", block.device, block.variant)
        os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
        plan_path = os.path.join(out_dir, "raw", f"plan_{block.block_id}.json")
        with open(plan_path, "w") as fh:
            json.dump({"settings": st.as_dict(), "cells": [c.as_dict() for c in block.cells], "cell_limit_s": cell_cap}, fh)
        # GPU conclusions need STRICT placement (soft placement could hide CPU fallbacks), so a strict failure is reported
        # as an unsupported configuration instead of being silently re-run in soft mode.
        placement = "soft" if args.placement == "soft" else "strict"
        res = run_job(args, limits, out_dir, job, placement, remaining_s, ("--plan-path", plan_path), block.block_id)
        return {"status": res.get("status"), "reason": res.get("reason", ""), "cells": res.get("data", {}).get("cells", []),
                "placement": placement, "notes": res.get("notes", []), "data": res.get("data", {})}

    hw = sysinfo.hardware_info()
    headroom = memory_headroom_gb(limits, take_sample())
    print(f"scaling: memory headroom for planning = {headroom:.1f} GB (configurations estimated above {headroom / 1.5:.1f} GB are skipped)", flush=True)
    job = scaling_run.run_scaling(st, headroom, runner, {"chip": hw.get("chip"), "ram_gb": hw.get("ram_total_gb"),
                                                                    "software": sysinfo.package_versions()})
    interrupted = job.pop("_interrupted", None)
    if interrupted is not None:
        job["_reraise"] = interrupted
    return job


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    limits = limits_from(args)
    if args.worker:
        return worker_main(args, limits)
    if args.dry_run:
        return dry_run(args, limits)

    errs = validate_args(args)
    if errs:
        print("Invalid configuration:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    pre = preflight(limits, take_sample())
    if "scaling" in args.tests:
        pre += scaling_preflight(args, limits, take_sample())
    if pre:
        print("Refusing to start - resource limits are already (nearly) violated:\n  " + "\n  ".join(pre) +
              "\nClose other applications and retry. Limits are never raised automatically.", file=sys.stderr)
        return 2

    from . import report, sysinfo
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    started = _dt.datetime.now(_dt.timezone.utc)
    print(f"tfbench {__version__} | results -> {out_dir}")
    hw = sysinfo.hardware_info()
    meta = {
        "run_id": os.path.basename(os.path.normpath(out_dir)), "started_utc": started.isoformat(timespec="seconds"),
        "command": " ".join([os.path.basename(SCRIPT), *(argv if argv is not None else sys.argv[1:])]),
        "hardware": hw, "software": sysinfo.package_versions(), "limits": limits.as_dict(),
        "power_source": sysinfo.power_source(), "low_power_mode": sysinfo.low_power_mode(),
        "thermal_before": sysinfo.thermal_state(),
        "config": {k: v for k, v in vars(args).items()
                   if k not in ("worker", "run_placement", "result_path", "budget_s", "max_swap_gb")},
    }
    jobs, interrupted = supervise(args, limits, out_dir)
    for j in jobs:
        j.pop("_reraise", None)
    meta["finished_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    meta["thermal_after"] = sysinfo.thermal_state()
    meta["interrupted"] = interrupted
    rep = report.build_report(meta, jobs)
    paths = report.write_reports(rep, out_dir)
    for j in jobs:
        if j.get("test") == "scaling" and j.get("data", {}).get("analysis"):
            from .scaling import report as scaling_report
            paths["scaling"] = scaling_report.write_artifacts(out_dir, j)["summary"]
    print("\nReports written:")
    for k, v in paths.items():
        print(f"  {k:<9} {v}")
    for j in jobs:
        if j.get("test") == "scaling" and j.get("data", {}).get("controller"):
            ctl = j["data"]["controller"]
            if ctl["skipped"] or j["status"] != "completed":
                from collections import Counter
                reasons = Counter(x["reason"][:90] for x in ctl["skipped"])
                print(f"\nscaling: status {j['status']}" + (f" - {j['reason']}" if j.get("reason") else "") + f"; {len(ctl['skipped'])} configuration(s) skipped:")
                for r, n in reasons.most_common(5):
                    print(f"  {n:>3} x {r}")
    bad = [j for j in jobs if j.get("status") != "completed"]
    print(f"\n{len(jobs) - len(bad)}/{len(jobs)} job(s) completed." + (" Incomplete: " + ", ".join(
        f"{j['job']} ({j['status']})" for j in bad) if bad else ""))
    return 1 if bad else 0


# ----------------------------------------------------------------------- worker
def worker_main(args, limits: ResourceLimits) -> int:
    job = Job.parse(args.worker)
    rec = Recorder(job.name, args.result_path)
    placement = args.run_placement
    rec.data["worker"] = {"pid": os.getpid(), "placement": placement, "budget_s": args.budget_s}
    monitor: Optional[ResourceMonitor] = None
    try:
        # Import TensorFlow BEFORE taking the memory baseline: the import footprint is not "test" memory.
        from . import tfenv
        if not (job.test == "scalingblock" and (job.variant or "").startswith("mp")):
            tfenv.import_tf()                 # multiprocessing blocks own a pool of spawned workers and never import TF themselves
        monitor = ResourceMonitor(limits, interval=0.5, runtime_budget_s=args.budget_s, hard_grace_s=30.0)

        def hard_stop(reason: str) -> None:       # watchdog thread, worker unresponsive for 30 s
            rec.set_status("aborted", f"{reason} (hard stop: worker did not react within 30 s)")
            rec.note("Worker was blocked (likely inside a native TensorFlow/Metal call) when the limit fired.")
            try:
                rec.write_final(monitor)
            finally:
                os._exit(70)

        monitor.on_hard_stop = hard_stop
        signal.signal(signal.SIGTERM, lambda *_: monitor.request_abort("terminated by supervisor"))
        monitor.start()
        _dispatch(args, rec, monitor, job, placement, limits)
        if rec.status == "running":
            rec.status = "completed"
        if monitor.aborted and rec.status == "completed":
            rec.set_status("aborted", monitor.reason)
    except LimitExceeded as e:
        rec.set_status("aborted", e.reason)
        print(f"  ABORTED: {e.reason}", flush=True)
    except KeyboardInterrupt:
        rec.set_status("interrupted", "KeyboardInterrupt")
    except BaseException as e:                      # noqa: BLE001
        from . import tfenv
        name = type(e).__name__
        if name == "DeviceUnavailable":
            rec.set_status("unsupported", str(e))
        elif tfenv.is_placement_error(e):
            rec.set_status("placement_failed", f"{name}: {str(e)[:600]}")
            rec.note("An op requested on the GPU has no GPU kernel (unsupported on this TensorFlow/Metal build).")
        else:
            rec.set_status("failed", f"{name}: {str(e)[:300]}")
            rec.data["traceback"] = traceback.format_exc()[-3000:]
        print(f"  {rec.status.upper()}: {rec.reason}", flush=True)
    finally:
        if monitor is not None:
            monitor.stop()
        try:
            rec.write_final(monitor)
        except Exception as e:                      # pragma: no cover
            print(f"could not write result: {e}", file=sys.stderr)
    return 0 if rec.status == "completed" else 1


def _dispatch(args, rec: Recorder, monitor: ResourceMonitor, job: Job, placement: str, limits: ResourceLimits) -> None:
    from . import tfenv
    if job.test in LEGACY:
        from . import legacy
        tf = tfenv.import_tf()
        if args.cpu:
            tf.config.set_visible_devices([], "GPU")
        legacy.init(limits, args.budget_s, tf)
        print(f"TF {tf.__version__} | devices: {[d.name for d in tf.config.list_physical_devices()]} | "
              f"limits: +{limits.max_extra_gb:.1f} GB, >= {limits.min_avail_gb:.1f} GB free, "
              f"< {limits.max_sys_pct:.0f}% used, swap < {limits.max_swap_mb:.0f} MB", flush=True)
        legacy.status("start")
        getattr(legacy, f"test_{job.test}")(args, rec)
        return
    if job.test == "performance":
        from . import perf
        print(f"performance | {job.device} | {job.variant} | placement={placement} | mode={args.perf_mode}", flush=True)
        perf.run(args, rec, monitor, job.device, job.variant, placement)
        if rec.status == "running":
            print(f"  done: {_headline(rec)}", flush=True)
        return
    if job.test == "scalingblock":
        from .scaling import execute
        print(f"scaling block | {job.device} | {job.variant} | placement={placement}", flush=True)
        execute.run_block(args, rec, monitor, job, placement, args.plan_path)
        return
    from . import evolution
    if job.test == "neuroevolution":
        print(f"neuroevolution | {job.device} | placement={placement} | pop={args.ne_pop} gens={args.ne_generations}",
              flush=True)
        evolution.run(args, rec, monitor, job.device, placement)
    elif job.test == "memdiag":
        print(f"memdiag | {job.device} | policy={job.variant} | cycles={args.memdiag_cycles}", flush=True)
        evolution.run_memdiag(args, rec, monitor, job.device, placement, job.variant)


def _headline(rec: Recorder) -> str:
    for n in ("steady_samples_per_s",):
        for m in rec.measurements:
            if m["name"] == n:
                return f"{n} = {m['value']:,.0f}"
    return "ok"
