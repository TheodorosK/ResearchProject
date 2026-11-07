"""SYNTHETIC cell generator with a KNOWN ground-truth cost model.

For tests and for illustrating the report layout ONLY.  It never runs in a real benchmark and its numbers say nothing about
any real machine.  Because the truth is known, tests can check that the adaptive design and the crossover analysis recover it.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from .workload import Cell, Point, Settings, Strategy, elite_count, n_evaluations

# (fixed seconds per evaluation, seconds per training step) per (device, tier): GPU = high fixed / low per-step cost
TRUTH: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("cpu", "small"): (0.10, 0.0004), ("cpu", "medium"): (0.12, 0.0010), ("cpu", "large"): (0.20, 0.0300),
    ("gpu", "small"): (1.20, 0.0025), ("gpu", "medium"): (1.25, 0.0030), ("gpu", "large"): (1.30, 0.0040),
}
FIXED_SPLIT = {"construct": 0.25, "trace": 0.40, "first_exec": 0.30, "fitness": 0.05}


def true_break_even(tier: str) -> Optional[float]:
    fc, sc = TRUTH[("cpu", tier)]
    fg, sg = TRUTH[("gpu", tier)]
    return (fg - fc) / (sc - sg) if sc > sg and fg > fc else None


def synth_cell(cell: Cell, st: Settings, rng: random.Random, noise: float = 0.03,
               batched_gain: float = 2.0, mp_eff: float = 0.75, placement: str = "strict") -> Dict[str, Any]:
    pt, s = cell.point, cell.strategy
    fixed, step = TRUTH[(s.device, pt.tier)]
    e = n_evaluations(pt.population, pt.generations)
    jitter = lambda: math.exp(rng.gauss(0, noise))
    comps = {k: e * fixed * f * jitter() for k, f in FIXED_SPLIT.items()}
    comps["first_exec"] += e * step
    comps["train"] = e * (pt.steps - 1) * step * jitter()
    comps["ga"], comps["cleanup"] = 0.01 * pt.generations, 0.02 * pt.generations
    work = sum(comps.values())
    if s.mode == "batched":
        work /= batched_gain if pt.mode == "homo" else 1.0 + 0.4 * (batched_gain - 1.0)
    wall = work / (s.workers * mp_eff) if s.mode == "mp" else work
    wall *= 1.0
    if s.mode == "mp":
        scale = wall / work if work else 1.0
    tier_mem = {"small": 0.002, "medium": 0.01, "large": 0.08}[pt.tier]
    add_gb = (0.9 if s.device == "gpu" else 0.1) + (pt.population * tier_mem if s.mode == "batched" else tier_mem)
    gens = [{"gen": g, "evaluated": e // pt.generations, "wall_s": wall / pt.generations, "ga_s": 0.01, "cleanup_s": 0.02,
             "best_val_r2": 0.5 + 0.1 * g} for g in range(pt.generations)]
    return {"cell_id": cell.cell_id, "point": pt.as_dict(), "strategy": {"device": s.device, "mode": s.mode, "workers": s.workers},
            "phase": cell.phase, "rep": cell.rep, "status": "completed", "reason": "", "placement": placement,
            "evaluations": e, "failed_evaluations": 0, "wall_s": wall, "generations_run": pt.generations, "gens": gens,
            "comps": {**comps, "unattributed": (wall - sum(comps.values())) if s.mode != "mp" else None},
            "comps_basis": "worker-seconds (summed over workers)" if s.mode == "mp" else "wall-seconds",
            "ips": e / wall, "steps_per_s": e * pt.steps / wall,
            "train_steps_per_s": (e * (pt.steps - 1) / comps["train"]) if comps["train"] > 0 else None,
            "best_val_r2": 0.5 + 0.1 * pt.generations, "peak_additional_gb": add_gb, "peak_extra_gb_abs": add_gb + 0.1,
            "min_avail_gb": 6.0, "max_sys_pct": 60.0, "peak_rss_gb": 1.0, "swap_growth_mb": 0.0, "retained_gb": 0.02,
            "batched_member_fraction": (1.0 if pt.mode == "homo" else 0.4) if s.mode == "batched" else None,
            "padded_ratio_mean": 1.3 if (s.mode == "batched" and pt.mode == "hetero") else (1.0 if s.mode == "batched" else None),
            "train_fraction": (comps["train"] + comps["first_exec"]) / wall if wall else None,
            "signature": f"{pt.key}", "mp": ({"workers": s.workers, "startup_s": 6.0, "threads_per_worker": 1, "parallel_efficiency": mp_eff}
                                              if s.mode == "mp" else None)}


def make_runner(st: Settings, clock_ref: List[float], seed: int = 0, aborts: Optional[Callable[[Cell], bool]] = None,
                unsupported_devices: Tuple[str, ...] = (), soft_gpu: bool = False, noise: float = 0.03,
                calls: Optional[List[str]] = None) -> Callable:
    """Runner for ``Controller``.  ``clock_ref`` is a one-element list holding the simulated clock (advanced here)."""
    rng = random.Random(seed)

    def runner(block, remaining_s: float) -> Dict[str, Any]:
        if calls is not None:
            calls.append(block.block_id)
        clock_ref[0] += 8.0                                              # process start + warm-up overhead
        if block.device in unsupported_devices:
            return {"status": "unsupported", "reason": f"no {block.device} visible", "cells": [], "placement": "strict"}
        out = []
        for c in block.cells:
            if aborts is not None and aborts(c):
                rec = synth_cell(c, st, rng, noise)
                rec.update(status="aborted", reason="per-configuration time limit reached", ips=None, steps_per_s=None)
                clock_ref[0] += st.max_cell_s
                out.append(rec)
                continue
            rec = synth_cell(c, st, rng, noise, placement="soft" if (soft_gpu and c.strategy.device == "gpu") else "strict")
            clock_ref[0] += rec["wall_s"] + (6.0 if c.strategy.mode == "mp" else 0.0)
            out.append(rec)
        return {"status": "completed", "reason": "", "cells": out, "placement": "soft" if (soft_gpu and block.device == "gpu") else "strict",
                "data": {"block": {"cold_start_s": 5.0}}}
    return runner
