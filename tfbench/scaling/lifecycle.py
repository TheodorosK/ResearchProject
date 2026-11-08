"""Memory-retention and process-lifecycle design (``--scaling-design lifecycle``).

Questions (each answered from measured snapshots, never from assumption):
  * Does memory held after a GPU-batched cell get reused by the next cell, stabilise, or keep growing?
  * How does a long-lived TensorFlow process compare with a fresh process per cell (same cells, same order)?
  * Which layer holds it: process RSS, process USS (private pages), the OS footprint (``footprint`` where available),
    or only the system-wide available-memory figure?
  * What causes the ~85 MB per resident medium-tier GPU model seen in the capacity run (variables only / trace only / full)?

Terminology (kept strict): "retained" = still held at a given moment; "returned" = given back to the OS; "reused" = a later cell
did not need new memory for the same work.  A LEAK is claimed only when retained memory keeps growing over repeated identical
rounds AND is not returned when the process exits; allocator caching or system-wide changes in available memory are not leaks.

Safety: a cumulative retention guard stops the long-lived sequence when the memory held since the runtime became ready would
exceed 60 % of the headroom or the next cell's projection (x1.5) would not fit; the existing watchdog limits apply unchanged.
"""
from __future__ import annotations

import gc
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

from ..limits import GB, LimitExceeded, take_sample
from ..sysinfo import swap_counters
from .controller import BlockSpec, Controller
from .workload import Cell, Point, Settings, Strategy

pc = time.perf_counter
CUM_GUARD_FRACTION = 0.60

_UNIT = {"B": 1.0, "KB": 1024.0, "MB": 1024.0 ** 2, "GB": 1024.0 ** 3}


# ------------------------------------------------------------------------------------------------ snapshots
def _size(txt: str) -> Optional[float]:
    m = re.match(r"\s*([\d.,]+)\s*([KMG]?B)\b", txt)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "")) * _UNIT[m.group(2)]
    except (ValueError, KeyError):
        return None


def parse_footprint(text: str) -> Dict[str, Any]:
    """Best-effort parse of macOS ``footprint <pid>`` output (format ASSUMED from documentation; the raw text of the first
    call is stored in the report so a wrong assumption is visible).  Returns total footprint and dirty bytes per category."""
    out: Dict[str, Any] = {"footprint_gb": None, "categories": {}}
    m = re.search(r"Footprint:\s*([\d.,]+\s*[KMG]?B)", text)
    if m:
        v = _size(m.group(1))
        out["footprint_gb"] = None if v is None else v / GB
    row = re.compile(r"^\s*([\d.,]+\s*[KMG]?B)\s+([\d.,]+\s*[KMG]?B)\s+([\d.,]+\s*[KMG]?B)\s+(\d+)\s+(\S.*?)\s*$")
    for line in text.splitlines():
        r = row.match(line)
        if r:
            dirty, clean = _size(r.group(1)), _size(r.group(2))
            if dirty is not None:
                out["categories"][r.group(5)] = {"dirty_gb": dirty / GB, "clean_gb": (clean or 0.0) / GB, "regions": int(r.group(4))}
    return out


_FIRST_FP_RAW: List[str] = []


def _footprint(pid: int) -> Tuple[Optional[float], Dict[str, Any]]:
    if sys.platform != "darwin":
        return None, {}
    try:
        txt = subprocess.run(["footprint", str(pid)], capture_output=True, text=True, timeout=10).stdout
    except Exception:                                                    # noqa: BLE001 - tool missing / not permitted
        return None, {}
    if not _FIRST_FP_RAW:
        _FIRST_FP_RAW.append(txt[:4000])
    parsed = parse_footprint(txt)
    return parsed["footprint_gb"], parsed["categories"]


def first_footprint_raw() -> Optional[str]:
    return _FIRST_FP_RAW[0] if _FIRST_FP_RAW else None


def proc_mem(pid: Optional[int] = None, want_footprint: bool = False) -> Dict[str, Any]:
    p = psutil.Process(pid)
    out: Dict[str, Any] = {"rss_gb": p.memory_info().rss / GB, "uss_gb": None, "footprint_gb": None}
    try:
        out["uss_gb"] = p.memory_full_info().uss / GB
    except Exception:                                                    # noqa: BLE001
        pass
    if want_footprint:
        out["footprint_gb"], out["footprint_categories"] = _footprint(p.pid)
    return out


def snapshot(label: str, footprint: bool = True) -> Dict[str, Any]:
    s = take_sample()
    sw = swap_counters()
    m = proc_mem(want_footprint=footprint)
    return {"label": label, "t": time.time(), "avail_gb": s.avail_gb, "sys_pct": s.sys_pct, "swap_used_gb": sw["used_gb"],
            "swap_out_gb": sw["sout_gb"], "rss_gb": m["rss_gb"], "uss_gb": m["uss_gb"], "footprint_gb": m["footprint_gb"],
            "footprint_categories": m.get("footprint_categories") or None}


# ------------------------------------------------------------------------------------------------- worker side
def run_instrumented(ctx, cells: List[Cell], extra: Dict[str, Any], rec, monitor, record_cell, cell_limit: float) -> None:
    """Run ``cells`` in THIS process, with snapshots before / right after / after gc / settled for every cell."""
    from .design import estimate_memory_gb
    from .execute import run_cell
    st = ctx.settings
    headroom = float(extra["headroom_gb"])
    settle = float(extra.get("settle_s", 3.0))
    out: Dict[str, Any] = {"label": extra.get("label", "long"), "cells": [], "stopped": None, "headroom_gb": headroom}
    rec.data["lifecycle"] = out
    ready = snapshot("runtime ready")
    out["runtime_ready"] = ready
    for cell in cells:
        now = take_sample()
        cum = ready["avail_gb"] - now.avail_gb
        need = estimate_memory_gb(cell.point, cell.strategy, st)
        if cum > CUM_GUARD_FRACTION * headroom or cum + 1.5 * need > headroom:
            out["stopped"] = (f"retention guard before {cell.cell_id}: {cum:.2f} GB already held since the runtime was ready and the next cell "
                              f"needs ~{need:.2f} GB (x1.5) of the {headroom:.2f} GB headroom; stopped instead of continuing")
            break
        pre = snapshot("before")
        cr = run_cell(ctx, cell, cell_limit)
        limit_hit = cr.pop("_limit", False)
        record_cell(rec, cr)
        post = snapshot("after execution")
        t = pc()
        gc.collect()
        gc_s = pc() - t
        time.sleep(1.0)
        after_gc = snapshot("after gc")
        time.sleep(settle)
        settled = snapshot("settled")
        out["cells"].append({"cell_id": cell.cell_id, "population": cell.point.population, "rep": cell.rep, "status": cr["status"], "gc_s": gc_s,
                             "ips": cr.get("ips"), "wall_s": cr.get("wall_s"), "peak_additional_gb": cr.get("peak_additional_gb"),
                             "peak_extra_block_gb": cr.get("peak_extra_gb_abs"), "min_avail_gb": cr.get("min_avail_gb"),
                             "snaps": {"before": pre, "after_exec": post, "after_gc": after_gc, "settled": settled}})
        rec.checkpoint()
        if limit_hit:
            raise LimitExceeded(cr["reason"])
        if cr["status"] != "completed":
            out["stopped"] = f"{cell.cell_id} ended '{cr['status']}': {cr.get('reason', '')[:120]}"
            break
    fp = first_footprint_raw()
    if fp:
        out["footprint_raw_sample"] = fp


# --------------------------------------------------------------------------------------------- controller
def lifecycle_cells(st: Settings) -> List[Cell]:
    return [Cell(Point("large", p, st.steps[-1] if st.steps else 20, "homo", st.generations), Strategy(st.lifecycle_device, "batched"), "lifecycle", rep=r)
            for r in range(st.lifecycle_rounds) for p in st.lifecycle_pops]


MEDIUM_VARIANTS = (("full", False, "full: build + trace + execute, heterogeneous"), ("full", True, "full: homogeneous (one shared architecture)"),
                   ("trace", False, "trace only (never executed)"), ("vars", False, "variables + optimiser state only (no graph)"))


def design_summary_lifecycle(st: Settings, headroom_gb: float) -> Dict[str, Any]:
    from . import design as D
    cells = lifecycle_cells(st)
    blocks = 1 + (len(cells) if st.lifecycle_fresh else 0) + (len(MEDIUM_VARIANTS) if st.lifecycle_medium else 0)
    secs = (len(cells) * 8.0 + 30.0) + (len(cells) * 35.0 if st.lifecycle_fresh else 0.0) + (len(MEDIUM_VARIANTS) * 60.0 if st.lifecycle_medium else 0.0)
    worst = max(D.estimate_memory_gb(c.point, c.strategy, st) for c in cells)
    return {"design": "lifecycle", "cells": len(cells), "blocks": blocks, "rounds": st.lifecycle_rounds, "pops": list(st.lifecycle_pops),
            "estimated_total_s": secs, "estimated_total_s_range": (0.6 * secs, 2.0 * secs), "budget_s": st.max_total_s,
            "per_cell_limit_s": st.max_cell_s, "worst_case_cell_memory_gb": worst,
            "cells_over_headroom_by_estimate": [c.cell_id for c in cells if D.estimate_memory_gb(c.point, c.strategy, st) * 1.5 > headroom_gb],
            "kind": "estimated"}


class LifecycleController(Controller):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.long: Optional[Dict[str, Any]] = None
        self.fresh: List[Dict[str, Any]] = []
        self.variants: List[Dict[str, Any]] = []
        self._n = 0

    def _block(self, variant: str, cells: List[Cell], extra: Dict[str, Any], tag: str) -> Dict[str, Any]:
        self._n += 1
        spec = BlockSpec(f"{tag}-{self._n}", self.st.lifecycle_device, variant, cells, "lifecycle", 0, extra=extra)
        self.log(f"  block {spec.block_id}: {len(cells)} cell(s) | elapsed {self.elapsed():.0f}s / {self.st.max_total_s:.0f}s")
        t = self.clock()
        res = self.runner(spec, max(self.remaining(), 30.0))
        status = res.get("status", "completed")
        self.blocks.append({"block_id": spec.block_id, "device": self.st.lifecycle_device, "variant": variant, "phase": "lifecycle", "rep": 0, "status": status,
                            "reason": res.get("reason", ""), "placement": res.get("placement"), "n_planned": len(cells),
                            "n_returned": len((res.get("data") or {}).get("cells", [])), "wall_s": self.clock() - t, "cold_start_s": None, "attempts": 1})
        self.notes.extend(n for n in res.get("notes", []) if n not in self.notes)
        for r in (res.get("data") or {}).get("cells", []):
            r.setdefault("phase", "lifecycle")
            self.results.append(r)
        if status in ("unsupported", "placement_failed", "failed"):
            self._mark_unusable(self.st.lifecycle_device, f"{self.st.lifecycle_device}-batched", status, res.get("reason", ""))
        return res

    def run(self) -> Dict[str, Any]:
        from . import design as D
        try:
            cells = lifecycle_cells(self.st)
            runnable = []
            for c in cells:
                why = D.skip_reason(c, self.st, self.cost, self.max_memory_gb, self.blocked)
                (self._skip(c, why) if why else runnable.append(c))
            if not runnable or self._unusable_reason(Strategy(self.st.lifecycle_device, "batched")):
                self.status = "completed"
                return self.result()
            base = {"headroom_gb": self.max_memory_gb, "settle_s": self.st.lifecycle_settle_s}
            res = self._block("lifecycle", runnable, {**base, "label": "long"}, "lifecycle-long")
            self.long = (res.get("data") or {}).get("lifecycle")
            if self.st.lifecycle_fresh:
                for c in runnable:
                    if self.remaining() < 90:
                        self._skip(c, "time budget exhausted before this fresh-process run")
                        continue
                    if self._unusable_reason(Strategy(self.st.lifecycle_device, "batched")):
                        self._skip(c, "strategy unavailable")
                        continue
                    time.sleep(2.0)                                     # the previous process may still be returning its memory
                    s0, sw0 = take_sample(), swap_counters()
                    res = self._block("lifecycle", [c], {**base, "label": "fresh"}, "lifecycle-fresh")
                    time.sleep(2.0)                                     # let the OS reclaim the exited process
                    s1, sw1 = take_sample(), swap_counters()
                    d = (res.get("data") or {}).get("lifecycle") or {}
                    d.update(cell_id=c.cell_id, population=c.point.population, rep=c.rep, status=res.get("status"),
                             supervisor={"avail_before_spawn_gb": s0.avail_gb, "avail_after_exit_gb": s1.avail_gb,
                                         "swap_before_gb": sw0["used_gb"], "swap_after_gb": sw1["used_gb"]})
                    self.fresh.append(d)
            if self.st.lifecycle_medium:
                for mode, homo, desc in MEDIUM_VARIANTS:
                    if self.remaining() < 120:
                        self._skip_variant(mode, homo, "time budget exhausted before this probe")
                        continue
                    res = self._block("resident", [], {"tier": "medium", "ladder": [1, 2, 4, 8, 16, 32], "headroom_gb": self.max_memory_gb,
                                                       "mode": mode, "homo": homo}, f"resident-medium-{mode}{'-homo' if homo else ''}")
                    data = (res.get("data") or {}).get("resident")
                    if data:
                        data["description"] = desc
                        self.variants.append(data)
                    else:
                        self._skip_variant(mode, homo, f"probe produced no data ({res.get('status')}: {res.get('reason', '')[:100]})")
            self.status = "completed"
        except BaseException as e:                                  # noqa: BLE001
            self.status, self.reason = "interrupted", f"{type(e).__name__}: {e}"
            raise
        finally:
            self.phase_log["total"] = {"elapsed_s": self.elapsed()}
        return self.result()

    def _skip_variant(self, mode: str, homo: bool, reason: str) -> None:
        self.skipped.append({"cell_id": f"lifecycle:resident-medium:{mode}{'-homo' if homo else ''}", "point": {"tier": "medium", "population": 32, "steps": 1,
                                                                                                               "mode": "homo" if homo else "hetero", "generations": 1},
                             "strategy": f"{self.st.lifecycle_device}-resident", "phase": "lifecycle", "rep": 0, "status": "skipped", "reason": reason})

    def result(self) -> Dict[str, Any]:
        r = super().result()
        r.update(lifecycle_long=self.long, lifecycle_fresh=self.fresh, lifecycle_variants=self.variants)
        return r


# ---------------------------------------------------------------------------------------------------- analysis
def _d(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


def long_rows(long: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per cell, relative to the moment the runtime was ready (positive = memory held)."""
    if not long or not long.get("runtime_ready"):
        return []
    r0 = long["runtime_ready"]
    rows = []
    for c in long["cells"]:
        sn = c["snaps"]
        row = {"cell_id": c["cell_id"], "population": c["population"], "round": c["rep"], "status": c["status"], "ips": c["ips"],
               "peak_additional_gb": c["peak_additional_gb"], "min_avail_gb": c["min_avail_gb"]}
        for k in ("before", "after_exec", "after_gc", "settled"):
            s = sn[k]
            row[k] = {"system": _d(r0["avail_gb"], s["avail_gb"]), "rss": _d(s["rss_gb"], r0["rss_gb"]), "uss": _d(s["uss_gb"], r0["uss_gb"]),
                      "footprint": _d(s["footprint_gb"], r0["footprint_gb"]), "swap_mb": _d(s["swap_used_gb"], r0["swap_used_gb"]) and 1024 * _d(s["swap_used_gb"], r0["swap_used_gb"])}
        rows.append(row)
    return rows


def classify_retention(rows: List[Dict[str, Any]], rounds: int, per_round: int) -> Dict[str, Any]:
    """Compare the memory held (system-wide, settled) at the end of each round.  Thresholds are stated in the text, not hidden."""
    ends = [r for r in rows if r["round"] is not None]
    by_round: Dict[int, float] = {}
    for r in ends:
        v = r["settled"]["system"]
        if v is not None:
            by_round[r["round"]] = v                                  # last cell of each round overwrites
    seq = [by_round[k] for k in sorted(by_round)]
    inc = [b - a for a, b in zip([0.0] + seq[:-1], seq)]
    out: Dict[str, Any] = {"held_at_end_of_round_gb": seq, "increment_per_round_gb": inc, "rounds_completed": len(seq), "rounds_planned": rounds}
    if len(seq) < 3:
        out.update(verdict="inconclusive", reason=f"only {len(seq)} complete round(s); at least 3 are needed to see whether growth stops")
        return out
    first, last = inc[0], inc[-1]
    tol = 0.15                                                         # GB: below the run-to-run noise seen in earlier sessions
    if abs(last) <= max(tol, 0.1 * abs(first)):
        out.update(verdict="stabilised", reason=f"the last round added {last:+.2f} GB versus {first:+.2f} GB in the first: later rounds reuse what is held (threshold: <= max(0.15 GB, 10% of round 1))")
    elif last >= 0.6 * first and first > tol:
        out.update(verdict="still_growing", reason=f"each round still adds about the same ({[round(x, 2) for x in inc]}); no plateau within {len(seq)} rounds")
    else:
        out.update(verdict="partly_reused", reason=f"increments shrink ({[round(x, 2) for x in inc]}) but do not reach the plateau threshold")
    return out


def fresh_rows(fresh: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for d in fresh:
        sv, rd = d.get("supervisor") or {}, d.get("runtime_ready")
        row = {"cell_id": d.get("cell_id"), "population": d.get("population"), "status": d.get("status")}
        if sv:
            row["net_held_after_exit_gb"] = _d(sv["avail_before_spawn_gb"], sv["avail_after_exit_gb"])
            row["swap_growth_mb"] = 1024 * (sv["swap_after_gb"] - sv["swap_before_gb"])
        if rd:
            row["runtime_ready_system_gb_below_spawn"] = _d(sv.get("avail_before_spawn_gb"), rd["avail_gb"]) if sv else None
            row["runtime_ready_rss_gb"], row["runtime_ready_uss_gb"] = rd["rss_gb"], rd["uss_gb"]
        if d.get("cells"):
            c = d["cells"][0]
            sn = c["snaps"]
            row.update(ips=c["ips"], peak_additional_gb=c["peak_additional_gb"], held_after_exec_gb=_d(sn["before"]["avail_gb"], sn["after_exec"]["avail_gb"]),
                       held_after_gc_gb=_d(sn["before"]["avail_gb"], sn["after_gc"]["avail_gb"]), held_settled_gb=_d(sn["before"]["avail_gb"], sn["settled"]["avail_gb"]),
                       rss_growth_gb=_d(sn["settled"]["rss_gb"], sn["before"]["rss_gb"]), uss_growth_gb=_d(sn["settled"]["uss_gb"], sn["before"]["uss_gb"]))
        rows.append(row)
    return rows


def variant_rows(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from .capacity import fit_resident
    out = []
    for v in variants:
        f = fit_resident(v)
        rungs = v.get("rungs") or []
        out.append({"description": v.get("description", v.get("mode")), "mode": v.get("mode"), "homo": v.get("homo"), "n_rungs": len(rungs),
                    "extra_at_max_gb": rungs[-1]["extra_gb"] if rungs else None, "max_resident": rungs[-1]["n_resident"] if rungs else 0,
                    "gb_per_model_system": f.get("gb_per_model"), "gb_per_model_uss": f.get("uss_gb_per_model"),
                    "gb_per_model_footprint": f.get("footprint_gb_per_model"), "intercept_gb": f.get("intercept_gb"),
                    "stopped": v.get("stopped"), "release": v.get("release")})
    return out


def analyze_lifecycle(res: Dict[str, Any], st: Settings, headroom_gb: float) -> Dict[str, Any]:
    long = res.get("lifecycle_long")
    rows = long_rows(long)
    return {"headroom_gb": headroom_gb, "long_rows": rows, "long_stopped": (long or {}).get("stopped"),
            "retention": classify_retention(rows, st.lifecycle_rounds, len(st.lifecycle_pops)), "fresh_rows": fresh_rows(res.get("lifecycle_fresh", [])),
            "variants": variant_rows(res.get("lifecycle_variants", [])), "footprint_raw_sample": (long or {}).get("footprint_raw_sample"),
            "runtime_ready_long": (long or {}).get("runtime_ready")}


def _f(v, nd=2, unit=""):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def markdown(an: Dict[str, Any], st: Settings, status: str, elapsed_s: float, skipped: List[Dict[str, Any]]) -> str:
    L = ["# Memory retention and process lifecycle", "",
         f"Status **{status}**, elapsed {elapsed_s / 60:.1f} min of {st.max_total_s / 60:.0f}. GPU-batched homogeneous large tier, P={list(st.lifecycle_pops)}, "
         f"{st.lifecycle_rounds} identical rounds, S={st.steps[-1] if st.steps else 20}, G={st.generations}, batch {st.batch}. Headroom {an['headroom_gb']:.1f} GB. "
         "All memory values are GB relative to the moment the TensorFlow runtime was ready (after validation and warm-up); positive = held.", "",
         "## 1. Long-lived process: memory held after each cell (MEASURED)", "",
         "| round | P | ind/s | before | after execution | after gc | settled (+4 s) | process RSS | process USS | OS footprint | swap MB |", "|" + "---|" * 11]
    for r in an["long_rows"]:
        L.append(f'| {r["round"]} | {r["population"]} | {_f(r["ips"])} | {_f(r["before"]["system"])} | {_f(r["after_exec"]["system"])} | {_f(r["after_gc"]["system"])} | '
                 f'{_f(r["settled"]["system"])} | {_f(r["settled"]["rss"])} | {_f(r["settled"]["uss"])} | {_f(r["settled"]["footprint"])} | {_f(r["settled"]["swap_mb"], 0)} |')
    if an.get("long_stopped"):
        L += ["", f"**Sequence stopped:** {an['long_stopped']}"]
    ret = an["retention"]
    L += ["", f"**Retention behaviour (system-wide, settled, end of each round):** {[round(x, 2) for x in ret['held_at_end_of_round_gb']]} GB; increments {[round(x, 2) for x in ret['increment_per_round_gb']]}. "
          f"Verdict: **{ret['verdict']}** - {ret['reason']}.", "",
          "Reading rules: *stabilised* = later rounds reuse held memory (consistent with allocator caching); *still growing* = candidate for a leak, which additionally requires that "
          "the memory is not returned when the process exits (section 2); system-wide available memory also moves with other processes, so differences below ~0.15 GB are noise. Allocator caching and changes in system-wide available memory are not, by themselves, evidence of a leak.", "",
          "## 2. Fresh process per cell (MEASURED): what is held inside, and what the OS gets back on exit", "",
          "| cell | status | ind/s | peak additional | held after exec | after gc | settled | RSS growth | USS growth | net held after process exit | swap MB |", "|" + "---|" * 11]
    for r in an["fresh_rows"]:
        L.append(f'| {r["cell_id"]} | {r["status"]} | {_f(r.get("ips"))} | {_f(r.get("peak_additional_gb"))} | {_f(r.get("held_after_exec_gb"))} | {_f(r.get("held_after_gc_gb"))} | '
                 f'{_f(r.get("held_settled_gb"))} | {_f(r.get("rss_growth_gb"))} | {_f(r.get("uss_growth_gb"))} | {_f(r.get("net_held_after_exit_gb"))} | {_f(r.get("swap_growth_mb"), 0)} |')
    L += ["", "A net value near zero after exit means the OS reclaimed everything the process had (cached or retained memory is bounded by the process lifetime). "
          "Fresh processes pay the runtime start-up cost again (compare ind/s and the construction share).", ""]
    rd = an.get("runtime_ready_long")
    if rd:
        L += [f"Runtime-ready baseline of the long-lived process: RSS {_f(rd['rss_gb'])} GB, USS {_f(rd['uss_gb'])} GB, OS footprint {_f(rd['footprint_gb'])} GB.", ""]
    L += ["## 3. Medium-tier GPU memory per resident model: what causes it? (MEASURED probes, fresh process each, models 1..32)", "",
          "| variant | models | additional at max (system) | system GB/model | process USS GB/model | OS footprint GB/model | intercept GB |", "|---|---|---|---|---|---|---|"]
    for v in an["variants"]:
        L.append(f'| {v["description"]} | {v["max_resident"]} | {_f(v["extra_at_max_gb"])} | {_f(v["gb_per_model_system"], 3)} | {_f(v["gb_per_model_uss"], 3)} | '
                 f'{_f(v["gb_per_model_footprint"], 3)} | {_f(v["intercept_gb"])} |')
    L += ["", "Interpretation key: if the system figure per model is large for *variables only* the cost is in allocating variables/optimiser state; if it appears only with *trace* or *execute* "
          "it is graph/kernel state; if it is much larger for heterogeneous than homogeneous architectures it is per-shape kernel/pipeline state; if system GB/model greatly exceeds process "
          "USS GB/model the memory is outside the process's private pages (GPU driver / shared with the OS) and RSS/USS would understate it.", ""]
    if an.get("footprint_raw_sample"):
        L += ["## 4. Raw `footprint` output (first sample; format-assumption check)", "", "```", an["footprint_raw_sample"][:1500], "```", ""]
    L += ["## 5. Skipped or incomplete", ""]
    sk = [s for s in skipped]
    L += [f'* {s["cell_id"]}: {s["reason"]}' for s in sk] or ["* none"]
    return "\n".join(L) + "\n"
