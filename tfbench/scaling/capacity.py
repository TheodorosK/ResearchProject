"""Memory-capacity and concurrency design (``--scaling-design capacity``).

Question: how much unified memory does this workload actually need, and which resource limits it?  Four readings are kept
APART so that none is blamed for another's cost:

  * memory capacity   - GB per simultaneously RESIDENT model (resident probe: models are built and KEPT; ladder 1,2,4,...),
                        fixed runtime overhead, peak/headroom use of every grid cell, memory-driven skips;
  * compute           - does throughput keep rising with population/batch (more arithmetic in flight) or plateau;
  * TF-Metal overhead - per-model fixed cost and per-step dispatch cost that do not depend on the size of the work;
  * construction/cleanup - time to build/trace/first-execute a model, teardown share, memory NOT returned after release.

Sequential population size (P models evaluated one after another, each freed) is NOT concurrently resident memory:
only the probe and the batched / multiprocessing strategies keep several models alive at once.

Safety: the probe projects the memory of every rung from what it has measured and refuses a rung whose projection exceeds
90 % of the headroom (recorded as skipped with the reason); after the probe, the measured per-parameter slope vetoes batched
grid cells whose projected memory (x1.25) would not fit.  Nothing here raises a limit or provokes an out-of-memory condition.
"""
from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import tfenv
from ..evolution import build_keras
from ..limits import GB, LimitExceeded, capture_baseline, extra_gb, take_sample
from ..sysinfo import swap_counters
from .controller import BlockSpec, Controller
from .workload import (Cell, Point, Settings, Strategy, TIERS, batch_indices, dataset_for, individual_seed, initial_population,
                       params_of)

pc = time.perf_counter
PROJECTION_FRACTION = 0.90        # a rung / cell may only be run if its projected peak is below this share of the headroom
BATCHED_MARGIN = 1.5            # same margin as the closed-form estimate; applied to max(estimate, measured projection)


# ------------------------------------------------------------------------------------------------ grid design
def capacity_cells(st: Settings) -> Dict[str, List[Cell]]:
    """Fixed (non-adaptive) grid grouped by priority: batched (memory-hungry, informative) -> sequential -> multiprocessing."""
    out: Dict[str, List[Cell]] = {"batched": [], "seq": [], "mp": []}
    for tier in st.tiers:
        for steps in st.steps:
            for pop in st.pops:
                het, hom = Point(tier, pop, steps, "hetero", st.generations), Point(tier, pop, steps, "homo", st.generations)
                for dev in ("gpu", "cpu"):
                    if not st.skip_batched:
                        out["batched"].append(Cell(hom, Strategy(dev, "batched"), "capacity"))
                    out["seq"].append(Cell(het, Strategy(dev, "seq"), "capacity"))
                if not st.skip_mp:
                    for w in st.workers:
                        if w <= st.max_workers:
                            out["mp"].append(Cell(het, Strategy("cpu", "mp", w), "capacity"))
    return out


def theory_gb(params: float, st: Settings, tier: str) -> float:
    """Weights + gradient + Adam m,v in float32 (16 B/param) + activations of one training step."""
    t = TIERS[tier]
    act = st.batch * max(t.units) * (t.max_layers + 1) * 4 * 3
    return (params * 16 + act) / GB


def design_summary_capacity(st: Settings, headroom_gb: float) -> Dict[str, Any]:
    from . import design as D
    cost = D.CostModel()
    cells = [c for grp in capacity_cells(st).values() for c in grp]
    secs = sum(cost.predict(c.point, c.strategy, st) for c in cells)
    resident_blocks = 2 * len(st.resident_tiers)
    n_models = sum(st.ladder)
    resident_s = resident_blocks * (D.PRIORS["block_overhead_s"] + 0.5 * n_models * 1.2)
    variants = {(c.strategy.device, c.strategy.variant) for c in cells}
    overhead = (len(variants) + resident_blocks) * D.PRIORS["block_overhead_s"]
    total = secs + resident_s + overhead
    worst = max([D.estimate_memory_gb(c.point, c.strategy, st) for c in cells] or [0.0])
    skipped_pre = [c for c in cells if D.estimate_memory_gb(c.point, c.strategy, st) * 1.5 > headroom_gb]
    return {"design": "capacity", "grid_cells": len(cells), "resident_blocks": resident_blocks, "resident_ladder": list(st.ladder),
            "worker_processes_max": len(variants) + resident_blocks, "estimated_total_s": total,
            "estimated_total_s_range": (0.5 * total, 2.0 * total), "budget_s": st.max_total_s, "per_cell_limit_s": st.max_cell_s,
            "worst_case_cell_memory_gb": worst, "cells_over_headroom_by_estimate": [c.cell_id for c in skipped_pre], "kind": "estimated"}


# -------------------------------------------------------------------------------------------- resident probe
def _build_resident(tf, device: str, genome, data, st: Settings, mode: str = "full") -> Dict[str, Any]:
    """mode: ``full`` build + trace + first execution | ``trace`` build + trace, never executed | ``vars`` variables and optimiser state only."""
    seed = individual_seed(st.seed, genome.uid)
    with tfenv.device_scope(tf, device):
        model = build_keras(tf, genome, st.in_dim, seed)
        tvars = model.trainable_variables
        opt, _ = tfenv.make_optimizer(tf, genome.learning_rate, tvars, "simple")
        idx = batch_indices(seed, len(data["x_train"]), st.batch, 2)
        xb, yb = tf.constant(data["x_train"][idx]), tf.constant(data["y_train"][idx])
        tfenv.sync(tvars[-1], xb, yb)
        if mode == "vars":
            return {"model": model, "opt": opt, "step": None, "xb": xb, "yb": yb, "tvars": tvars}

        @tf.function(jit_compile=False)
        def step(x, y):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(y - tf.squeeze(model(x, training=True), -1)))
            opt.apply_gradients(zip(tape.gradient(loss, tvars), tvars))
            return loss

        if mode == "trace":
            step.get_concrete_function(tf.TensorSpec((st.batch, st.in_dim), tf.float32), tf.TensorSpec((st.batch,), tf.float32))
            return {"model": model, "opt": opt, "step": None, "xb": xb, "yb": yb, "tvars": tvars}
        loss = step(xb[0], yb[0])
        tfenv.sync(loss, tvars[-1])
    return {"model": model, "opt": opt, "step": step, "xb": xb, "yb": yb, "tvars": tvars}


def _proc_fields() -> Dict[str, Any]:
    from .lifecycle import proc_mem
    m = proc_mem(want_footprint=True)
    return {"uss_gb": m.get("uss_gb"), "footprint_gb": m.get("footprint_gb")}


def _project_gb(rungs: List[Dict[str, Any]], cur_extra: float, new_params: List[int], st: Settings, tier: str, device: str) -> float:
    """Projected additional memory after adding models with ``new_params`` parameters (conservative)."""
    add_theory = sum(theory_gb(p, st, tier) for p in new_params)
    if len(rungs) < 2:
        # nothing measured yet: runtime overhead prior + 3x the theoretical model memory
        from . import design as D
        return cur_extra + (D.PRIORS["gpu_runtime_gb" if device == "gpu" else "cpu_runtime_gb"] if not rungs else 0.0) + 3.0 * add_theory
    a, b = rungs[-2], rungs[-1]
    dp = b["total_params"] - a["total_params"]
    slope = max((b["extra_gb"] - a["extra_gb"]) / dp, 0.0) if dp > 0 else 0.0
    return cur_extra + max(1.3 * slope * sum(new_params), 1.0 * add_theory)


def run_resident(ctx, extra: Dict[str, Any], rec, monitor) -> None:
    """Worker side: keep ever more models alive, measure memory after each rung, then release and measure what comes back."""
    st, tf, device = ctx.settings, ctx.tf, ctx.device
    tf.get_logger().setLevel("ERROR")                    # one tf.function per resident model is intended: silence the retracing warning
    tier, headroom = extra["tier"], float(extra["headroom_gb"])
    ladder = sorted({int(n) for n in extra["ladder"] if int(n) >= 1})
    mode, homo = extra.get("mode", "full"), bool(extra.get("homo", False))
    genomes = initial_population(Point(tier, max(ladder), 1, "homo" if homo else "hetero", 1), st)
    params = [params_of(g, st) for g in genomes]
    data = ctx.data
    base = monitor.baseline if monitor is not None and getattr(monitor, "baseline", None) is not None else capture_baseline()
    sw0 = swap_counters()
    out: Dict[str, Any] = {"tier": tier, "device": device, "ladder": ladder, "headroom_gb": headroom, "mode": mode, "homo": homo, "rungs": [], "stopped": None,
                           "release": None, "mean_params": float(np.mean(params)), "kind": "measured"}
    rec.data["resident"] = out
    held: List[Dict[str, Any]] = []
    ctx.deadline = time.monotonic() + 0.8 * ctx.settings.max_cell_s
    if monitor is not None:
        monitor.window_peaks()
    try:
        for target in ladder:
            n_have = len(held)
            if target <= n_have:
                continue
            new_p = params[n_have:target]
            cur = extra_gb(base, take_sample())
            proj = _project_gb(out["rungs"], cur, new_p, st, tier, device)
            if proj > PROJECTION_FRACTION * headroom:
                out["stopped"] = (f"rung of {target} resident models not attempted: projected additional memory {proj:.2f} GB exceeds "
                                  f"{PROJECTION_FRACTION:.0%} of the {headroom:.2f} GB headroom (projection from measured growth)")
                break
            t0 = pc()
            for i in range(n_have, target):
                ctx.check()
                held.append(_build_resident(tf, device, genomes[i], data, st, mode))
            build_s = pc() - t0
            t0 = pc()
            if mode == "full":
                for h in held:                                     # one more training step on EVERY resident model
                    loss = h["step"](h["xb"][1], h["yb"][1])
                tfenv.sync(loss, held[-1]["tvars"][-1])
            step_s = pc() - t0
            peaks = monitor.window_peaks() if monitor is not None else {}
            s = take_sample()
            tot = int(sum(params[:target]))
            out["rungs"].append({
                "n_resident": target, "total_params": tot, "extra_gb": extra_gb(base, s),
                "peak_extra_gb": (peaks.get("max_extra_gb") if peaks else None), "rss_gb": s.rss_gb, "avail_gb": s.avail_gb,
                "min_avail_gb": (peaks.get("min_avail_gb") if peaks else None),
                "swap_growth_mb": (s.swap_used_gb - sw0["used_gb"]) * 1024, "theory_gb": sum(theory_gb(p, st, tier) for p in params[:target]),
                **_proc_fields(),
                "build_s_added": build_s, "build_s_per_model": build_s / (target - n_have),
                "step_ms_per_model": (1000.0 * step_s / target) if mode == "full" else None,
                "projected_gb": proj})
            rec.checkpoint()
    except (LimitExceeded, Exception) as e:                        # noqa: BLE001 - CellTimeout / limit: keep what was measured
        if isinstance(e, LimitExceeded) or tfenv.is_placement_error(e):
            out["stopped"] = f"stopped by the watchdog: {getattr(e, 'reason', e)}"
            _release(out, held, base, sw0)
            rec.checkpoint()
            raise
        out["stopped"] = f"{type(e).__name__}: time limit or error - {str(e)[:120]}"
    _release(out, held, base, sw0)
    rec.checkpoint()


def _release(out: Dict[str, Any], held: List[Dict[str, Any]], base, sw0) -> None:
    peak = max([r["extra_gb"] for r in out["rungs"]] or [0.0])
    n = len(held)
    s_hold = take_sample()
    pm_hold = _proc_fields()
    held.clear()
    s_refs = take_sample()
    t = pc()
    gc.collect()
    gc_s = pc() - t
    time.sleep(1.0)
    s_gc = take_sample()
    time.sleep(2.0)
    s_late = take_sample()
    pm_late = _proc_fields()
    out["release"] = {"proc_before": pm_hold, "proc_after_3s": pm_late, "n_released": n, "peak_extra_gb": peak, "extra_before_release_gb": extra_gb(base, s_hold),
                      "extra_after_dropping_refs_gb": extra_gb(base, s_refs), "extra_after_gc_gb": extra_gb(base, s_gc),
                      "extra_after_3s_gb": extra_gb(base, s_late), "gc_s": gc_s,
                      "returned_fraction": ((peak - extra_gb(base, s_late)) / peak) if peak > 0 else None,
                      "swap_growth_mb": (s_late.swap_used_gb - sw0["used_gb"]) * 1024}


def fit_resident(r: Dict[str, Any]) -> Dict[str, Any]:
    """Least squares extra_gb = intercept + slope * total_params (and per model).  Needs >= 3 rungs."""
    rungs = r.get("rungs") or []
    res: Dict[str, Any] = {"device": r["device"], "tier": r["tier"], "n_rungs": len(rungs), "max_resident": max([x["n_resident"] for x in rungs] or [0]),
                           "kind": "calculated"}
    if len(rungs) >= 3:
        x = np.array([q["total_params"] for q in rungs], float)
        y = np.array([q["extra_gb"] for q in rungs], float)
        n = np.array([q["n_resident"] for q in rungs], float)
        slope, icpt = np.polyfit(x, y, 1)
        slope_n, icpt_n = np.polyfit(n, y, 1)
        pred = icpt + slope * x
        ss_res, ss_tot = float(np.sum((y - pred) ** 2)), float(np.sum((y - y.mean()) ** 2))
        theory = sum(q["theory_gb"] for q in rungs) / max(sum(q["total_params"] for q in rungs), 1)
        uss = [q.get("uss_gb") for q in rungs]
        if all(u is not None for u in uss):
            res["uss_gb_per_model"] = float(np.polyfit(n, np.array(uss, float), 1)[0])
        fp = [q.get("footprint_gb") for q in rungs]
        if all(u is not None for u in fp):
            res["footprint_gb_per_model"] = float(np.polyfit(n, np.array(fp, float), 1)[0])
        res.update(gb_per_param=float(slope), gb_per_model=float(slope_n), intercept_gb=float(icpt), intercept_per_model_fit_gb=float(icpt_n),
                   bytes_per_param=float(slope * GB), r2=(1 - ss_res / ss_tot) if ss_tot > 0 else None,
                   theory_bytes_per_param=float(theory * GB), overhead_factor=float(slope / theory) if theory > 0 else None)
    return res


# ----------------------------------------------------------------------------------------------- controller
class CapacityController(Controller):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.resident: List[Dict[str, Any]] = []
        self.fits: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._n_res = 0

    # -- guard from measured slope
    def run_cells(self, cells: List[Cell], phase: str, rep: int, cap_s: float) -> None:
        keep = []
        for c in cells:
            f = self.fits.get((c.strategy.device, c.point.tier))
            if c.strategy.mode == "batched" and f and f.get("gb_per_param") is not None:
                from .workload import point_param_stats
                total = point_param_stats(c.point, self.st)["mean"] * c.point.population
                from .design import estimate_memory_gb
                proj = max(max(f["intercept_gb"], 0.0) + f["gb_per_param"] * total, estimate_memory_gb(c.point, c.strategy, self.st))
                if proj * BATCHED_MARGIN > self.max_memory_gb:
                    self._skip(c, f"memory: projected {proj:.2f} GB (max of closed-form estimate and the slope measured by the resident probe, "
                                  f"x{BATCHED_MARGIN} margin) exceeds the {self.max_memory_gb:.1f} GB headroom")
                    continue
            keep.append(c)
        super().run_cells(keep, phase, rep, cap_s)

    def _resident(self, dev: str, tier: str) -> None:
        name = f"{dev}-resident"
        why = self._unusable_reason(Strategy(dev, "seq")) or self.unusable.get(name)
        if why:
            self._skip_resident(dev, tier, f"strategy unavailable: {why}")
            return
        if self.remaining() < 120:
            self._skip_resident(dev, tier, "time budget exhausted before this probe")
            return
        self._n_res += 1
        spec = BlockSpec(f"resident-{dev}-{tier}-{self._n_res}", dev, "resident", [], "capacity", 0,
                         extra={"tier": tier, "ladder": list(self.st.ladder), "headroom_gb": self.max_memory_gb})
        self.log(f"  block {spec.block_id}: resident probe, ladder {list(self.st.ladder)} | elapsed {self.elapsed():.0f}s / {self.st.max_total_s:.0f}s")
        t = self.clock()
        res = self.runner(spec, max(self.remaining(), 30.0))
        data = (res.get("data") or {}).get("resident")
        status = res.get("status", "completed")
        self.blocks.append({"block_id": spec.block_id, "device": dev, "variant": "resident", "phase": "capacity", "rep": 0, "status": status,
                            "reason": res.get("reason", ""), "placement": res.get("placement"), "n_planned": 0, "n_returned": 0,
                            "wall_s": self.clock() - t, "cold_start_s": None, "attempts": 1})
        self.notes.extend(n for n in res.get("notes", []) if n not in self.notes)
        if status in ("unsupported", "placement_failed", "failed"):
            self._mark_unusable(dev, name, status, res.get("reason", ""))
        if status in ("crashed", "killed", "aborted"):
            self.notes.append(f"{spec.block_id} ended '{status}': {res.get('reason', '')[:160]}")
        if data:
            data["block_status"], data["block_reason"] = status, res.get("reason", "")
            self.resident.append(data)
            f = fit_resident(data)
            if f.get("gb_per_param") is not None and status == "completed":
                self.fits[(dev, tier)] = f
        else:
            self._skip_resident(dev, tier, f"probe produced no data ({status}: {res.get('reason', '')[:100]})")

    def _skip_resident(self, dev: str, tier: str, reason: str) -> None:
        self.skipped.append({"cell_id": f"capacity:resident:{tier}:{dev}", "point": {"tier": tier, "population": max(self.st.ladder), "steps": 1,
                                                                                   "mode": "hetero", "generations": 1},
                             "strategy": f"{dev}-resident", "phase": "capacity", "rep": 0, "status": "skipped", "reason": reason})

    def run(self) -> Dict[str, Any]:
        try:
            tiers = list(self.st.resident_tiers)
            for tier in tiers[:1]:                                   # the decision-relevant tier first, on both devices
                for dev in ("gpu", "cpu"):
                    self._resident(dev, tier)
            grid = capacity_cells(self.st)
            for key in ("batched", "seq", "mp"):
                if grid[key] and self.remaining() > 60:
                    self.run_cells(grid[key], "capacity", 0, self.remaining())
                else:
                    for c in grid[key]:
                        self._skip(c, "time budget exhausted")
            for tier in tiers[1:]:
                for dev in ("gpu", "cpu"):
                    self._resident(dev, tier)
            self.status = "completed"
        except BaseException as e:                                  # noqa: BLE001
            self.status, self.reason = "interrupted", f"{type(e).__name__}: {e}"
            raise
        finally:
            self.phase_log["total"] = {"elapsed_s": self.elapsed()}
        return self.result()

    def result(self) -> Dict[str, Any]:
        r = super().result()
        r["resident"] = self.resident
        return r


# --------------------------------------------------------------------------------------------------- analysis
def _row(c: Dict[str, Any]) -> Dict[str, Any]:
    p, s = c["point"], c["strategy"]
    name = f'{s["device"]}-{s["mode"]}' + (str(s["workers"]) if s["mode"] == "mp" else "")
    comps = c.get("comps") or {}
    wall = c.get("wall_s") or 0.0
    ev = c.get("evaluations") or 0
    return {"strategy": name, "tier": p["tier"], "population": p["population"], "steps": p["steps"], "mode": p["mode"], "status": c["status"],
            "reason": c.get("reason", ""), "ips": c.get("ips"), "wall_s": c.get("wall_s"), "evaluations": ev,
            "peak_additional_gb": c.get("peak_additional_gb"), "peak_extra_block_gb": c.get("peak_extra_gb_abs"), "retained_rss_gb": c.get("retained_gb"),
            "retained_sys_gb": c.get("retained_sys_gb"), "swap_growth_mb": c.get("swap_growth_mb"), "min_avail_gb": c.get("min_avail_gb"),
            "fixed_share": ((sum(comps.get(k, 0.0) or 0.0 for k in ("construct", "trace", "first_exec", "teardown")) / wall)
                            if (wall and s["mode"] == "seq") else None),
            "teardown_share": ((comps.get("teardown", 0.0) or 0.0) / wall) if (wall and s["mode"] == "seq") else None,
            "retained_sys_mb_per_eval": (1024 * c["retained_sys_gb"] / ev) if (ev and c.get("retained_sys_gb") is not None) else None}


def capacity_table(cells: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [_row(c) for c in cells if c.get("point") and c.get("strategy")]
    skipped = [x for x in skipped if not str(x["strategy"]).endswith("-resident")]      # probe skips are listed separately
    done = {(r["strategy"], r["tier"], r["population"], r["steps"]) for r in rows}
    for s in skipped:
        p = s["point"]
        key = (s["strategy"], p["tier"], p["population"], p["steps"])
        if key not in done:
            rows.append({"strategy": s["strategy"], "tier": p["tier"], "population": p["population"], "steps": p["steps"], "mode": p.get("mode"),
                         "status": "skipped", "reason": s["reason"]})
    order = {"gpu-batched": 0, "cpu-batched": 1, "gpu-seq": 2, "cpu-seq": 3}
    return sorted(rows, key=lambda r: (r["tier"], order.get(r["strategy"], 9), r["strategy"], r["steps"], r["population"]))


def capacity_table_with_ratios(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seq = {(r["strategy"], r["tier"], r["steps"], r["population"]): r["ips"] for r in rows if r["status"] == "completed" and r.get("ips")}
    for r in rows:
        r["vs_seq_same_device"] = None
        if r["status"] == "completed" and r["strategy"].endswith(("batched", "mp2", "mp4")) and r.get("ips"):
            dev = r["strategy"].split("-")[0]
            base = seq.get((f"{dev}-seq", r["tier"], r["steps"], r["population"]))
            if base:
                r["vs_seq_same_device"] = r["ips"] / base
    return rows


def readings(rows: List[Dict[str, Any]], fits: List[Dict[str, Any]], resident: List[Dict[str, Any]], headroom_gb: float,
             skipped: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evidence tables for the four candidate limits.  These are indicators with stated thresholds, not verdicts."""
    ok = [r for r in rows if r["status"] == "completed"]
    peaks = [r["peak_additional_gb"] for r in ok if r.get("peak_additional_gb") is not None]
    mem_skips = [s for s in skipped if s["reason"].startswith(("estimated memory", "memory:", "rung")) or "headroom" in s["reason"]]
    cap = {"headroom_gb": headroom_gb, "largest_peak_additional_gb": max(peaks) if peaks else None,
           "largest_peak_share_of_headroom": (max(peaks) / headroom_gb) if (peaks and headroom_gb) else None,
           "max_swap_growth_mb": max([r["swap_growth_mb"] for r in ok if r.get("swap_growth_mb") is not None] or [0.0]),
           "cells_skipped_for_memory": len(mem_skips),
           "resident_stopped": [f'{q["device"]}/{q["tier"]}: {q["stopped"]}' for q in resident if q.get("stopped")]}
    comp: Dict[str, Any] = {}
    for dev in ("gpu", "cpu"):
        b = sorted([(r["population"], r["ips"]) for r in ok if r["strategy"] == f"{dev}-batched" and r["tier"] == "large"])
        if len(b) >= 2:
            last, prev = b[-1][1], b[-2][1]
            comp[dev] = {"batched_ips_by_P": b, "last_step_gain": last / prev if prev else None,
                         "reading": ("throughput still rising with population at the largest tested P (>= 1.15x for the last doubling): "
                                     "more arithmetic in flight still helps, i.e. not yet saturated") if prev and last / prev >= 1.15 else
                                    "throughput plateaued at the largest tested P (< 1.15x for the last doubling): saturated or limited by something other than parallelism"}
    ovh: Dict[str, Any] = {}
    for f in fits:
        if f.get("intercept_gb") is not None:
            ovh[f'{f["device"]}/{f["tier"]}'] = {"fixed_runtime_gb": f["intercept_gb"], "bytes_per_param_measured": f["bytes_per_param"],
                                                 "bytes_per_param_theory": f["theory_bytes_per_param"], "overhead_factor": f["overhead_factor"]}
    steps_ms = {f'{q["device"]}/{q["tier"]}': [(r["n_resident"], r["step_ms_per_model"]) for r in q["rungs"]] for q in resident if q["rungs"]}
    cons = {"seq_fixed_share_of_wall": {r["strategy"] + f'/P{r["population"]}': r["fixed_share"] for r in ok if r.get("fixed_share") is not None and r["tier"] == "large"},
            "seq_retained_sys_mb_per_eval": {r["strategy"] + f'/P{r["population"]}': r["retained_sys_mb_per_eval"] for r in ok
                                            if r.get("retained_sys_mb_per_eval") is not None and r["tier"] == "large"},
            "build_s_per_model": {f'{q["device"]}/{q["tier"]}': [(r["n_resident"], r["build_s_per_model"]) for r in q["rungs"]] for q in resident if q["rungs"]},
            "released": {f'{q["device"]}/{q["tier"]}': q["release"] for q in resident if q.get("release")}}
    return {"memory_capacity": cap, "compute": comp, "framework_overhead": {"resident_fixed_and_slope": ovh, "step_ms_per_resident_model": steps_ms},
            "construction_cleanup": cons}


def capacity_projection(fits: List[Dict[str, Any]], budgets_gb: Tuple[float, ...] = (6.0, 12.0, 24.0, 48.0, 72.0)) -> List[Dict[str, Any]]:
    """CALCULATED (not measured beyond the probe): resident models of the tier's mean size that fit a given amount of FREE memory,
    assuming the measured linear growth continues.  Says nothing about speed or about what else the machine needs."""
    out = []
    for f in fits:
        if f.get("gb_per_model") and f["gb_per_model"] > 0:
            out.append({"device": f["device"], "tier": f["tier"], "gb_per_model": f["gb_per_model"], "intercept_gb": f["intercept_gb"],
                        "tested_up_to": f["max_resident"], "kind": "calculated",
                        "models_fitting": {str(int(b)): max(0, int((b - max(f["intercept_gb"], 0.0)) / f["gb_per_model"])) for b in budgets_gb}})
    return out


def analyze_capacity(ctrl_result: Dict[str, Any], st: Settings, headroom_gb: float) -> Dict[str, Any]:
    rows = capacity_table_with_ratios(capacity_table(ctrl_result["cells"], ctrl_result["skipped"]))
    resident = ctrl_result.get("resident", [])
    fits = [fit_resident(q) for q in resident]
    return {"table": rows, "resident": resident, "resident_fits": fits, "readings": readings(rows, fits, resident, headroom_gb, ctrl_result["skipped"]),
            "projection_calculated": capacity_projection(fits), "headroom_gb": headroom_gb}


# --------------------------------------------------------------------------------------------------- report
def _f(v, nd=2, unit=""):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def markdown(cap: Dict[str, Any], st: Settings, status: str, elapsed_s: float, skipped: Optional[List[Dict[str, Any]]] = None) -> str:
    L = ["# Memory-capacity and concurrency benchmark", "",
         f"Status **{status}**, elapsed {elapsed_s / 60:.1f} min of a {st.max_total_s / 60:.0f} min budget. Planning headroom {cap['headroom_gb']:.1f} GB. "
         f"Batch {st.batch}, S={list(st.steps)}, G={st.generations}. Every number is labelled measured / calculated / estimated; nothing is extrapolated to other chips.", "",
         "## 1. Resident-model probe (MEASURED): memory held by simultaneously alive models", ""]
    L += ["| device/tier | resident models | total params (M) | additional GB (system) | peak GB | theory GB (16 B/param) | build s/model | step ms/model | swap MB |", "|---|---|---|---|---|---|---|---|---|"]
    for q in cap["resident"]:
        for r in q["rungs"]:
            L.append(f'| {q["device"]}/{q["tier"]} | {r["n_resident"]} | {r["total_params"] / 1e6:.1f} | {_f(r["extra_gb"])} | {_f(r["peak_extra_gb"])} | '
                     f'{_f(r["theory_gb"])} | {_f(r["build_s_per_model"])} | {_f(r["step_ms_per_model"], 1)} | {_f(r["swap_growth_mb"], 0)} |')
        if q.get("stopped"):
            L.append(f'| {q["device"]}/{q["tier"]} | - | - | - | - | - | - | - | **stopped: {q["stopped"]}** |')
    L += ["", "### Fits (CALCULATED from the rungs: additional GB = intercept + slope x parameters)", "",
          "| device/tier | rungs | intercept GB (fixed runtime) | GB per resident model | bytes/param measured | bytes/param theory | measured/theory | R2 |", "|---|---|---|---|---|---|---|---|"]
    for f in cap["resident_fits"]:
        L.append(f'| {f["device"]}/{f["tier"]} | {f["n_rungs"]} | {_f(f.get("intercept_gb"))} | {_f(f.get("gb_per_model"), 3)} | {_f(f.get("bytes_per_param"), 1)} | '
                 f'{_f(f.get("theory_bytes_per_param"), 1)} | {_f(f.get("overhead_factor"), 2, "x")} | {_f(f.get("r2"), 3)} |')
    L += ["", "### After release (MEASURED): memory returned to the system", "",
          "| device/tier | models released | peak GB | after dropping refs | after gc | after 3 s | returned fraction |", "|---|---|---|---|---|---|---|"]
    for q in cap["resident"]:
        r = q.get("release")
        if r:
            L.append(f'| {q["device"]}/{q["tier"]} | {r["n_released"]} | {_f(r["peak_extra_gb"])} | {_f(r["extra_after_dropping_refs_gb"])} | {_f(r["extra_after_gc_gb"])} | '
                     f'{_f(r["extra_after_3s_gb"])} | {_f(r["returned_fraction"], 2)} |')
    L += ["", "## 2. Strategy grid (MEASURED): population size vs resident memory and throughput", "",
          "Sequential strategies keep ONE model alive at a time, whatever P is; batched keeps P alive together; multiprocessing keeps one per worker.", "",
          "| strategy | tier | P | S | status | ind/s | vs seq (same device) | wall s | peak additional GB | peak vs block start GB | retained GB (RSS) | retained GB (system) | swap MB | min avail GB |", "|" + "---|" * 14]
    for r in cap["table"]:
        if r["status"] == "completed":
            L.append(f'| {r["strategy"]} | {r["tier"]} | {r["population"]} | {r["steps"]} | ok | {_f(r["ips"])} | {_f(r.get("vs_seq_same_device"), 2, "x")} | {_f(r["wall_s"], 1)} | '
                     f'{_f(r["peak_additional_gb"])} | {_f(r.get("peak_extra_block_gb"))} | {_f(r["retained_rss_gb"], 3)} | {_f(r["retained_sys_gb"], 3)} | {_f(r["swap_growth_mb"], 0)} | {_f(r["min_avail_gb"])} |')
        else:
            L.append(f'| {r["strategy"]} | {r["tier"]} | {r["population"]} | {r["steps"]} | **{r["status"]}** | - | - | - | - | - | - | - | - | {r.get("reason", "")[:110]} |')
    rd = cap["readings"]
    c = rd["memory_capacity"]
    L += ["", "## 3. What limits the workload? (indicators; thresholds stated, not verdicts)", "",
          "### 3a. Memory capacity",
          f"* Largest peak additional memory of any completed cell: {_f(c['largest_peak_additional_gb'])} GB = {_f(c['largest_peak_share_of_headroom'] and 100 * c['largest_peak_share_of_headroom'], 0)}% of the "
          f"{c['headroom_gb']:.1f} GB headroom; max swap growth {c['max_swap_growth_mb']:.0f} MB; configurations skipped for memory: {c['cells_skipped_for_memory']}.",
          "* Reading: memory capacity limited this run only where configurations were skipped or a probe stopped (listed in section 5); a machine with more memory could run those. "
          "If nothing was skipped for memory, the 16 GB machine was not memory-limited for the tested configurations."]
    for s in c["resident_stopped"]:
        L.append(f"* Probe stopped: {s}")
    L += ["", "### 3b. Compute (does throughput keep rising with more work in flight?)"]
    for dev, d in rd["compute"].items():
        L.append(f"* {dev} batched large tier, ind/s by P: " + ", ".join(f"P{p}={v:.2f}" for p, v in d["batched_ips_by_P"]) + f". {d['reading']}.")
    if not rd["compute"]:
        L.append("* not enough batched points completed")
    L += ["", "### 3c. TensorFlow-Metal / framework overhead (cost that does not depend on the amount of arithmetic)"]
    for k, v in rd["framework_overhead"]["resident_fixed_and_slope"].items():
        L.append(f"* {k}: fixed runtime memory {v['fixed_runtime_gb']:.2f} GB; {v['bytes_per_param_measured']:.1f} bytes/param measured vs {v['bytes_per_param_theory']:.1f} theory "
                 f"({v['overhead_factor']:.2f}x).")
    for k, v in rd["framework_overhead"]["step_ms_per_resident_model"].items():
        L.append(f"* {k}: ms per training step per resident model by count: " + ", ".join(f"{n}={ms:.1f}" for n, ms in v) +
                 ". Flat values mean per-step cost does not depend on how many models are alive (dispatch-bound, not memory-bound).")
    L += ["", "### 3d. Construction and cleanup inefficiency"]
    for k, v in rd["construction_cleanup"]["build_s_per_model"].items():
        L.append(f"* {k}: build + trace + first execution, s per model by rung: " + ", ".join(f"{n}={s:.2f}" for n, s in v))
    for k, v in sorted(rd["construction_cleanup"]["seq_fixed_share_of_wall"].items()):
        L.append(f"* sequential {k}: fixed per-model share of wall time (construct+trace+first_exec+teardown) {100 * v:.0f}%.")
    for k, v in sorted(rd["construction_cleanup"]["seq_retained_sys_mb_per_eval"].items()):
        L.append(f"* sequential {k}: system memory still held after the cell, {v:.1f} MB per evaluation (a value growing with P would indicate leaked models).")
    L += ["", "## 4. CALCULATED: resident models that fit in a given amount of FREE memory (linear continuation of the probe; memory only, no speed information)", "",
          "| device/tier | GB per model | fixed GB | probed up to | " + " | ".join(f"{b} GB" for b in ("6", "12", "24", "48", "72")) + " |", "|---|---|---|---|---|---|---|---|---|"]
    for p in cap["projection_calculated"]:
        L.append(f'| {p["device"]}/{p["tier"]} | {p["gb_per_model"]:.3f} | {p["intercept_gb"]:.2f} | {p["tested_up_to"]} | ' +
                 " | ".join(str(p["models_fitting"][k]) for k in ("6", "12", "24", "48", "72")) + " |")
    L += ["", "Values above the probed count are extrapolations of the memory fit only. Free memory is what is left after macOS and other applications.", "",
          "## 5. Skipped or incomplete configurations (never counted as measurements)", ""]
    bad = [r for r in cap["table"] if r["status"] != "completed"]
    for r in bad:
        L.append(f'* {r["strategy"]} {r["tier"]} P={r["population"]} S={r["steps"]}: **{r["status"]}** - {r.get("reason", "")}')
    for s_ in (skipped or []):
        if s_["strategy"].endswith("-resident"):
            L.append(f'* {s_["strategy"]} {s_["point"]["tier"]}: **skipped** - {s_["reason"]}')
    if not bad and not any(s_["strategy"].endswith("-resident") for s_ in (skipped or [])):
        L.append("* none")
    return "\n".join(L) + "\n"
