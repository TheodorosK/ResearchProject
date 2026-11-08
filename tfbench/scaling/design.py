"""Adaptive experimental design.  Pure Python: no TensorFlow, no I/O, fully unit-testable.

Phases
  pilot         sparse paired CPU/GPU sweep (tier x steps at one population size), 1 repetition each (exploration)
  expansion     around the most informative pilot points: other population sizes, multiprocessing, batching
  confirmation  independent repetitions (fresh processes) of the crossover candidates -> confidence intervals

Only confirmation data is used to declare a verdict; exploration data only selects what to confirm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import stats
from .workload import (TIER_ORDER, TIERS, Cell, Point, Settings, Strategy, elite_count, n_evaluations,
                       point_param_stats)

GB = 1024 ** 3

# --- planning PRIORS (used only for dry-run estimates and budgeting before measurements exist) -------------------------
# Derived from the author's M1 Pro baseline (GPU: ~1.4-1.8 s per freshly built model incl. 160 small steps; CPU ~0.15 s).
# They are replaced by measured values (CostModel.update) as soon as the pilot has run. Treat as +-2x.
PRIORS = {
    "cpu": {"fixed_base": 0.10, "fixed_per_layer": 0.010, "step_floor": 4e-4, "flops_per_s": 1.5e11},
    "gpu": {"fixed_base": 0.90, "fixed_per_layer": 0.050, "step_floor": 2.5e-3, "flops_per_s": 6e11},
    "ga_per_generation_s": 0.03, "mp_efficiency": 0.75, "mp_startup_s": 8.0, "block_overhead_s": 10.0,
    "gpu_runtime_gb": 0.9, "cpu_runtime_gb": 0.1, "process_footprint_gb": 0.7, "batched_coverage_hetero": 0.3,
}


@dataclass
class CostModel:
    """seconds per evaluation = fixed(device, tier) + steps * step(device, tier); priors until measurements arrive."""
    fitted: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    def fixed_step(self, device: str, tier: str, st: Settings) -> Tuple[float, float]:
        if (device, tier) in self.fitted:
            f = self.fitted[(device, tier)]
            return f["fixed"], f["step"]
        stat = point_param_stats(Point(tier, 4, 1, "hetero"), st)
        pr = PRIORS[device]
        fixed = pr["fixed_base"] + pr["fixed_per_layer"] * stat["layers_mean"]
        step = max(pr["step_floor"], 6.0 * stat["mean"] * st.batch / pr["flops_per_s"])
        return fixed, step

    def predict(self, point: Point, strategy: Strategy, st: Settings) -> float:
        fixed, step = self.fixed_step(strategy.device, point.tier, st)
        per_eval = fixed + point.steps * step
        e = n_evaluations(point.population, point.generations)
        ga = point.generations * PRIORS["ga_per_generation_s"]
        if strategy.mode == "seq":
            return e * per_eval + ga
        if strategy.mode == "mp":
            return e * per_eval / (strategy.workers * PRIORS["mp_efficiency"]) + ga
        # batched: one compile per bucket; homogeneous = 1 bucket, heterogeneous ~ low coverage
        if point.mode == "homo":
            gain = point.population ** (0.75 if strategy.device == "gpu" else 1.0)
            return point.generations * (fixed + point.steps * step * gain) + ga
        return e * per_eval * (1 - 0.5 * PRIORS["batched_coverage_hetero"]) + ga

    def update(self, results: Iterable[Dict[str, Any]], st: Settings) -> None:
        """Refit fixed/step per (device, tier) from completed sequential cells (medians over cells)."""
        acc: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
        for r in results:
            if r.get("status") != "completed" or r["strategy"]["mode"] != "seq" or not r.get("evaluations"):
                continue
            c, e, s = r["comps"], r["evaluations"], r["point"]["steps"]
            key = (r["strategy"]["device"], r["point"]["tier"])
            step = c["train"] / (e * (s - 1)) if s > 1 else None
            d = acc.setdefault(key, {"fixed": [], "step": []})
            if step is not None:
                d["step"].append(step)
            d["fixed_raw"] = d.get("fixed_raw", [])
            d["fixed_raw"].append((c["construct"] + c["trace"] + c["first_exec"] + c["fitness"] + c.get("teardown", 0.0)) / e)
        for key, d in acc.items():
            if d["step"] and d.get("fixed_raw"):
                step = stats.median_cv(d["step"])["median"]
                fixed = max(0.0, stats.median_cv(d["fixed_raw"])["median"] - step)   # first_exec contains one step
                self.fitted[key] = {"fixed": fixed, "step": step}


def estimate_memory_gb(point: Point, strategy: Strategy, st: Settings) -> float:
    """Closed-form ESTIMATE of additional memory for one cell (weights+grads+Adam moments = 16 B/param, activations,
    data, runtime overhead).  Used to refuse cells that could threaten the limits BEFORE running them."""
    stat = point_param_stats(point, st)
    t = TIERS[point.tier]
    widest = max(t.units)
    act = st.batch * widest * (t.max_layers + 1) * 4 * 3
    data = (st.train_n + 2 * st.val_n) * (st.in_dim + 1) * 4
    per_model = stat["max"] * 16 + act
    overhead = PRIORS["gpu_runtime_gb" if strategy.device == "gpu" else "cpu_runtime_gb"] * GB
    if strategy.mode == "batched":
        total = point.population * per_model * (1.3 if point.mode == "hetero" else 1.0) + data + overhead
    elif strategy.mode == "mp":
        total = strategy.workers * (PRIORS["process_footprint_gb"] * GB + per_model + data)
    else:
        total = per_model + data + overhead
    return total / GB


# ------------------------------------------------------------------------------------------- planning
def pilot_cells(st: Settings) -> List[Cell]:
    cells = []
    for device in ("cpu", "gpu"):
        for tier in st.tiers:
            for steps in st.pilot_steps:
                cells.append(Cell(Point(tier, st.pilot_pop, steps, "hetero", st.generations), Strategy(device), "pilot"))
    return cells


def _is(r: Dict[str, Any], mode: str, device: str, pop_mode: str = "hetero") -> bool:
    return (r["strategy"]["mode"] == mode and r["strategy"]["device"] == device and r["point"]["mode"] == pop_mode)


def _usable(r: Dict[str, Any]) -> bool:
    """Completed, positive throughput, and (for the GPU) strict placement: soft placement may hide CPU fallbacks."""
    return (r.get("status") == "completed" and (r.get("ips") or 0) > 0
            and not (r["strategy"]["device"] == "gpu" and r.get("placement") == "soft"))


def seq_ratio_table(results: Iterable[Dict[str, Any]], phases: Sequence[str] = ("pilot", "expansion")) -> List[Dict[str, Any]]:
    """GPU/CPU throughput ratio per hetero sequential point from exploration data (median over available reps)."""
    by: Dict[str, Dict[str, List[float]]] = {}
    pts: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if r["phase"] not in phases or not _usable(r) or r["strategy"]["mode"] != "seq" or r["point"]["mode"] != "hetero":
            continue
        key = Point(**r["point"]).key
        by.setdefault(key, {"cpu": [], "gpu": []})[r["strategy"]["device"]].append(r["ips"])
        pts[key] = r["point"]
    table = []
    for key, d in by.items():
        if d["cpu"] and d["gpu"]:
            c, g = stats.median_cv(d["cpu"])["median"], stats.median_cv(d["gpu"])["median"]
            table.append({"point": pts[key], "key": key, "ratio": g / c, "cpu_ips": c, "gpu_ips": g})
    return table


def choose_candidates(table: List[Dict[str, Any]], k: int, near: Tuple[float, float] = (0.5, 2.0)) -> List[Dict[str, Any]]:
    """Most informative points: the best GPU/CPU ratio; both sides of any sign change of (ratio-1) along steps or tier;
    then points near parity.  Always returns the best point even when the GPU is hopeless everywhere, so that a negative
    result is also confirmed rather than assumed."""
    if not table:
        return []
    chosen: List[Dict[str, Any]] = []

    def add(e):
        if all(e["key"] != c["key"] for c in chosen):
            chosen.append(e)

    ordered = sorted(table, key=lambda e: -e["ratio"])
    add(ordered[0])
    by_axis: Dict[Tuple, List[Dict[str, Any]]] = {}
    for e in table:
        p = e["point"]
        by_axis.setdefault(("steps", p["tier"], p["population"]), []).append(e)
        by_axis.setdefault(("tier", p["steps"], p["population"]), []).append(e)
    for (axis, *_), items in by_axis.items():
        keyf = (lambda e: e["point"]["steps"]) if axis == "steps" else (lambda e: TIER_ORDER.index(e["point"]["tier"]))
        items = sorted(items, key=keyf)
        for a, b in zip(items, items[1:]):
            if (a["ratio"] - 1) * (b["ratio"] - 1) < 0:
                add(a)
                add(b)
    for e in sorted(table, key=lambda e: abs(math.log(e["ratio"]))):
        if near[0] <= e["ratio"] <= near[1]:
            add(e)
    return chosen[:max(k, 1)]


def plan_expansion(cands: List[Dict[str, Any]], st: Settings, results: List[Dict[str, Any]], cost: CostModel,
                   budget_s: float, max_memory_gb: float, blocked: Optional[Dict[Tuple[str, str], float]] = None,
                   is_usable: Optional[Callable[[Strategy], bool]] = None) -> Tuple[List[Cell], List[Dict[str, Any]]]:
    """Returns (cells to run, skipped records).  Cells are ordered by information value and cut off by the time budget.
    Strategies that are known to be unavailable (``is_usable`` false) are dropped BEFORE budgeting."""
    blocked = blocked or {}
    is_usable = is_usable or (lambda s: True)
    done = {Cell(Point(**r["point"]), Strategy(**r["strategy"]), r["phase"], r["rep"]).cell_id.split(":", 1)[1]
            for r in results if r.get("status") in ("completed", "aborted", "failed")}
    prio: List[Tuple[int, Cell]] = []
    pmax = max(st.pops)

    def add(rank: int, point: Point, strategy: Strategy) -> None:
        prio.append((rank, Cell(point, strategy, "expansion")))

    for c in cands:
        p = c["point"]
        tier, steps, p0 = p["tier"], p["steps"], p["population"]
        base = Point(tier, p0, steps, "hetero", st.generations)
        others = sorted((q for q in st.pops if q != p0), key=lambda q: abs(q - p0))
        for i, pop in enumerate(others):                                           # nearest population first
            for dev in ("cpu", "gpu"):
                add(1 if i == 0 else 3, Point(tier, pop, steps, "hetero", st.generations), Strategy(dev))
        if not st.skip_batched:
            for dev in ("cpu", "gpu"):
                add(1, Point(tier, p0, steps, "homo", st.generations), Strategy(dev, "seq"))
                add(1, Point(tier, p0, steps, "homo", st.generations), Strategy(dev, "batched"))
                add(1, base, Strategy(dev, "batched"))
            if pmax != p0:
                for dev in ("cpu", "gpu"):
                    for mode in ("seq", "batched"):
                        add(4, Point(tier, pmax, steps, "homo", st.generations), Strategy(dev, mode))
        if not st.skip_mp:
            for w in st.workers:
                if w <= st.max_workers:
                    add(2, base, Strategy("cpu", "mp", w))
        if steps == max(st.steps) and st.extend_steps:                              # opt-in extension past the grid
            for s2 in st.extend_steps:
                for dev in ("cpu", "gpu"):
                    add(5, Point(tier, p0, s2, "hetero", st.generations), Strategy(dev))
    prio.sort(key=lambda rc: rc[0])                                                 # stable: keeps candidate order within a rank
    plan, skipped, spent, seen = [], [], 0.0, set()
    for _, cell in prio:
        sig = cell.cell_id.split(":", 1)[1]
        if sig in seen or sig in done:
            continue
        seen.add(sig)
        if not is_usable(cell.strategy):
            continue
        why = skip_reason(cell, st, cost, max_memory_gb, blocked)
        est = cost.predict(cell.point, cell.strategy, st)
        if why is None and spent + est > budget_s:
            why = "expansion time budget exhausted"
        if why:
            skipped.append({"cell_id": cell.cell_id, "reason": why})
        else:
            plan.append(cell)
            spent += est
    return plan, skipped


def skip_reason(cell: Cell, st: Settings, cost: CostModel, max_memory_gb: float,
                blocked: Dict[Tuple[str, str], float]) -> Optional[str]:
    est = cost.predict(cell.point, cell.strategy, st)
    if est > st.max_cell_s:
        return f"predicted {est:.0f}s exceeds the per-configuration limit of {st.max_cell_s:.0f}s"
    mem = estimate_memory_gb(cell.point, cell.strategy, st)
    if mem * 1.5 > max_memory_gb:
        return f"estimated memory {mem:.1f} GB (x1.5 margin) exceeds the {max_memory_gb:.1f} GB limit"
    floor = blocked.get((cell.strategy.name, cell.point.tier))
    if floor is not None and est >= floor:
        return "skipped: an equal or cheaper configuration of this strategy/tier already aborted"
    return None


def plan_confirmation(table: List[Dict[str, Any]], results: List[Dict[str, Any]], st: Settings) -> List[Tuple[Point, List[Strategy]]]:
    """Points to repeat (fresh processes) for CI-based verdicts.  Strategy-gain candidate (batched/mp) appended if found."""
    cands = choose_candidates(table, st.confirm_points)
    plan = [(Point(**c["point"]), [Strategy("cpu"), Strategy("gpu")]) for c in cands]
    gain = best_strategy_gain(results)
    if gain is not None:
        p, strategies = gain
        if all(p.key != q.key or {s.name for s in strategies} != {s.name for s in ss} for q, ss in plan):
            plan.append((p, strategies))
    return plan


def best_strategy_gain(results: List[Dict[str, Any]]) -> Optional[Tuple[Point, List[Strategy]]]:
    """Point where batching (either device) or CPU multiprocessing gave the largest exploration speed-up vs the
    sequential run of the identical Point."""
    seq: Dict[Tuple[str, str], float] = {}
    for r in results:
        if r.get("status") == "completed" and r["strategy"]["mode"] == "seq" and (r.get("ips") or 0) > 0:
            seq[(Point(**r["point"]).key, r["strategy"]["device"])] = r["ips"]
    best, best_gain = None, 1.15                                 # ignore <15 % gains: not worth the confirmation budget
    for r in results:
        if r.get("status") != "completed" or r["strategy"]["mode"] == "seq" or not _usable(r):
            continue
        base = seq.get((Point(**r["point"]).key, r["strategy"]["device"]))
        if base and r["ips"] / base > best_gain:
            best_gain = r["ips"] / base
            best = (Point(**r["point"]), [Strategy(r["strategy"]["device"], "seq"), Strategy(**r["strategy"])])
    return best


def confirmation_cells(plan: List[Tuple[Point, List[Strategy]]], st: Settings) -> List[List[Cell]]:
    """List per repetition; each repetition's cells are grouped into blocks by strategy by the controller."""
    reps = []
    for rep in range(st.confirm_reps):
        reps.append([Cell(p, s, "confirmation", rep) for p, ss in plan for s in ss])
    return reps


# ------------------------------------------------------------------------------------------- dry-run summary
def design_summary(st: Settings, cost: Optional[CostModel] = None) -> Dict[str, Any]:
    """Counts and PRIOR-based runtime/memory estimates for the planned design (upper bounds for cell counts)."""
    cost = cost or CostModel()
    pilot = pilot_cells(st)
    pilot_s = sum(cost.predict(c.point, c.strategy, st) for c in pilot)
    n_conf_points = st.confirm_points + (0 if (st.skip_batched and st.skip_mp) else 1)
    conf_cells = n_conf_points * 2 * st.confirm_reps
    typical = max(pilot, key=lambda c: cost.predict(c.point, c.strategy, st))
    # expansion: upper bound = candidates x (3 other pops x 2 devices + batched/homo 9 + P-max 4 + mp len(workers))
    per_cand = (len(st.pops) - 1) * 2 + (0 if st.skip_batched else 13) + (0 if st.skip_mp else len(st.workers))
    exp_cells = min(st.confirm_points, 2) * per_cand
    exp_s = 0.5 * exp_cells * (pilot_s / max(len(pilot), 1)) * 1.5
    conf_s = conf_cells * (pilot_s / max(len(pilot), 1)) * 1.2
    spawns = 2 + 8 + st.confirm_reps * (2 if (st.skip_batched and st.skip_mp) else 4)
    overhead = spawns * PRIORS["block_overhead_s"]
    worst_mem = max(estimate_memory_gb(c.point, c.strategy, st) for c in pilot)
    exp_probe = [Cell(Point(t, max(st.pops), max(st.steps), m, st.generations), Strategy(d, "batched"))
                 for t in st.tiers for m in ("hetero", "homo") for d in ("cpu", "gpu")]
    worst_mem = max(worst_mem, max(estimate_memory_gb(c.point, c.strategy, st) for c in exp_probe))
    return {
        "full_cartesian_points": len(st.tiers) * len(st.pops) * len(st.steps),
        "full_cartesian_cells_seq_only": len(st.tiers) * len(st.pops) * len(st.steps) * 2,
        "pilot_cells": len(pilot), "expansion_cells_max": exp_cells, "confirmation_cells": conf_cells,
        "total_cell_runs_max": len(pilot) + exp_cells + conf_cells, "worker_processes_max": spawns,
        "evaluations_per_pilot_cell": n_evaluations(st.pilot_pop, st.generations),
        "estimated_pilot_s": pilot_s, "estimated_total_s": pilot_s + exp_s + conf_s + overhead,
        "estimated_total_s_range": (0.5 * (pilot_s + exp_s + conf_s + overhead), 2.0 * (pilot_s + exp_s + conf_s + overhead)),
        "budget_s": st.max_total_s, "per_cell_limit_s": st.max_cell_s,
        "worst_case_cell_memory_gb": worst_mem, "kind": "estimated",
    }
