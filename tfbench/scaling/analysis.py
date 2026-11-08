"""Analysis of scaling results: verdicts, crossover, components, strategies, memory, cost model, answers.  Pure Python.

Rules:  verdicts come ONLY from confirmation repetitions (fresh processes); exploration data selects what to confirm and
is reported separately.  Aborted / skipped / failed configurations are listed and never enter any statistic.  Anything that
is a model or extrapolation is labelled "estimated"."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from . import design as D
from . import stats
from .workload import TIER_ORDER, Point, Settings, point_param_stats

COMP_ORDER = ("construct", "trace", "first_exec", "train", "fitness", "teardown", "ga", "cleanup", "unattributed")
COMPETITIVE = 0.5          # "competitive" = GPU reaches at least half of the CPU throughput (definition, not a measurement)


def _key(r: Dict[str, Any]) -> str:
    return Point(**r["point"]).key


def _med(xs: List[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None


def _usable(r: Dict[str, Any]) -> bool:
    return D._usable(r)


def _ips_lists(results, phases: Tuple[str, ...], mode: str = "seq", pop_modes: Tuple[str, ...] = ("hetero",)) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        if r["phase"] in phases and _usable(r) and r["strategy"]["mode"] == mode and r["point"]["mode"] in pop_modes:
            out.setdefault(_key(r), {"cpu": [], "gpu": []})[r["strategy"]["device"]].append(r["ips"])
    return out


def seq_points(results, st: Settings, mode: str = "seq", pop_modes: Tuple[str, ...] = ("hetero",)) -> List[Dict[str, Any]]:
    """CPU vs GPU for ONE identical strategy (``mode``) on identical workloads."""
    explore = _ips_lists(results, ("pilot", "expansion"), mode, pop_modes)
    conf = _ips_lists(results, ("confirmation",), mode, pop_modes)
    cells = {}
    for r in results:
        cells[_key(r)] = r["point"]
    explore = {k: v for k, v in explore.items() if k in cells}
    out = []
    for key in sorted(set(explore) | set(conf), key=lambda k: (TIER_ORDER.index(cells[k]["tier"]), cells[k]["population"], cells[k]["steps"])):
        p = cells[key]
        e, c = explore.get(key, {"cpu": [], "gpu": []}), conf.get(key, {"cpu": [], "gpu": []})
        e_ratio = (_med(e["gpu"]) / _med(e["cpu"])) if e["gpu"] and e["cpu"] else None
        ci = stats.log_ratio_ci(c["gpu"], c["cpu"]) if (c["gpu"] and c["cpu"]) else {"ratio": None, "lo": None, "hi": None, "n_num": 0, "n_den": 0}
        v = stats.verdict(ci) if ci["ratio"] is not None else ("unconfirmed" if e_ratio is not None else "no_data")
        out.append({"key": key, "tier": p["tier"], "population": p["population"], "steps": p["steps"],
                    "explore_ratio": e_ratio, "explore_cpu_ips": _med(e["cpu"]), "explore_gpu_ips": _med(e["gpu"]),
                    "confirm": {**ci, "cpu": stats.median_cv(c["cpu"]), "gpu": stats.median_cv(c["gpu"])},
                    "verdict": v, "ratio": ci["ratio"] if ci["ratio"] is not None else e_ratio,
                    "evidence": "confirmed (independent repetitions)" if ci.get("lo") is not None else ("exploration only (1 rep)" if e_ratio is not None else "none")})
    return out


def crossover(points: List[Dict[str, Any]], st: Settings) -> Dict[str, Any]:
    confirmed, candidates = [], []
    by_tier: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    by_steps: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for p in points:
        if p["ratio"] is None:
            continue
        by_tier.setdefault((p["tier"], p["population"]), []).append(p)
        by_steps.setdefault((p["steps"], p["population"]), []).append(p)
    for (tier, pop), items in by_tier.items():
        scan = stats.crossover_scan([{"x": i["steps"], "ratio": i["ratio"], "verdict": i["verdict"]} for i in sorted(items, key=lambda i: i["steps"])])
        for kind, lst in (("confirmed", confirmed), ("candidates", candidates)):
            lst.extend({"axis": "training steps", "tier": tier, "population": pop, **e} for e in scan[kind])
    for (steps, pop), items in by_steps.items():
        items = sorted(items, key=lambda i: TIER_ORDER.index(i["tier"]))
        scan = stats.crossover_scan([{"x": point_param_stats(Point(i["tier"], pop, steps), st)["mean"], "ratio": i["ratio"], "verdict": i["verdict"]} for i in items])
        for kind, lst in (("confirmed", confirmed), ("candidates", candidates)):
            lst.extend({"axis": "model size (mean parameters)", "steps": steps, "population": pop, **e} for e in scan[kind])
    with_ratio = [p for p in points if p["ratio"] is not None]
    best = max(with_ratio, key=lambda p: p["ratio"]) if with_ratio else None
    gpu_conf = [p["key"] for p in points if p["verdict"] == "gpu_faster"]
    cpu_conf = [p["key"] for p in points if p["verdict"] == "cpu_faster"]
    explored_gpu = [p["key"] for p in points if p["explore_ratio"] and p["explore_ratio"] > 1.0]
    status = "confirmed" if confirmed else ("candidate_unconfirmed" if (candidates or explored_gpu) else
                                            ("none_in_tested_range" if with_ratio else "no_paired_data"))
    return {"status": status, "confirmed_crossings": confirmed, "candidate_crossings": candidates,
            "gpu_faster_confirmed": gpu_conf, "cpu_faster_confirmed": cpu_conf, "gpu_faster_explored_only": [k for k in explored_gpu if k not in gpu_conf],
            "best": best, "n_points": len(points), "n_confirmed_points": sum(1 for p in points if p["confirm"].get("lo") is not None)}


def crossover_batched(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Crossover of the BATCHED strategy (GPU-batched vs CPU-batched) along population size, per (tier, steps, population mode)."""
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for p in points:
        if p["ratio"] is not None:
            groups.setdefault((p["tier"], p["steps"], p["key"].rsplit("/", 1)[-1]), []).append(p)
    confirmed, candidates = [], []
    for (tier, steps, pm), items in groups.items():
        scan = stats.crossover_scan([{"x": i["population"], "ratio": i["ratio"], "verdict": i["verdict"]}
                                     for i in sorted(items, key=lambda i: i["population"])])
        ctx = {"axis": "population size (batched strategy)", "tier": tier, "steps": steps, "population_mode": pm}
        confirmed.extend({**ctx, **e} for e in scan["confirmed"])
        candidates.extend({**ctx, **e} for e in scan["candidates"])
    return {"confirmed_crossings": confirmed, "candidate_crossings": candidates,
            "gpu_faster_confirmed": [p["key"] for p in points if p["verdict"] == "gpu_faster"],
            "cpu_faster_confirmed": [p["key"] for p in points if p["verdict"] == "cpu_faster"]}


def strategy_rows(results, st: Settings) -> List[Dict[str, Any]]:
    """Each non-sequential strategy vs the SEQUENTIAL run of the identical Point on the same device (cross-strategy)."""
    seq: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    other: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        if not _usable(r):
            continue
        k, d = _key(r), r["strategy"]["device"]
        grp = "conf" if r["phase"] == "confirmation" else "expl"
        if r["strategy"]["mode"] == "seq":
            seq.setdefault((k, d), {"conf": [], "expl": []})[grp].append(r["ips"])
        else:
            name = f'{d}-{r["strategy"]["mode"]}' + (str(r["strategy"]["workers"]) if r["strategy"]["mode"] == "mp" else "")
            e = other.setdefault((k, name), {"conf": [], "expl": [], "device": d, "point": r["point"], "cells": []})
            e[grp].append(r["ips"])
            e["cells"].append(r)
    rows = []
    for (k, name), e in sorted(other.items()):
        base = seq.get((k, e["device"]))
        if not base:
            continue
        use_conf = e["conf"] and base["conf"]
        a, b = (e["conf"], base["conf"]) if use_conf else (e["expl"] or e["conf"], base["expl"] or base["conf"])
        ci = stats.log_ratio_ci(a, b)
        c0 = e["cells"][0]
        rows.append({"point": k, "strategy": name, "device": e["device"], "population_mode": e["point"]["mode"],
                     "speedup_vs_sequential": ci["ratio"], "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                     "verdict": stats.verdict(ci) if use_conf else "unconfirmed",
                     "evidence": "confirmed" if use_conf and ci["lo"] is not None else "exploration (1 rep)",
                     "batched_member_fraction": _med([c["batched_member_fraction"] for c in e["cells"] if c.get("batched_member_fraction") is not None]),
                     "padded_ratio": _med([c["padded_ratio_mean"] for c in e["cells"] if c.get("padded_ratio_mean") is not None]),
                     "parallel_efficiency": (c0.get("mp") or {}).get("parallel_efficiency"),
                     "pool_startup_s": (c0.get("mp") or {}).get("startup_s")})
    return rows


def best_available(results) -> List[Dict[str, Any]]:
    """Best CPU strategy vs best GPU strategy per hetero Point (a CROSS-strategy comparison, labelled as such)."""
    best: Dict[str, Dict[str, Tuple[float, str]]] = {}
    measured: Dict[str, Dict[str, set]] = {}
    for r in results:
        if _usable(r) and r["phase"] != "confirmation":
            nm0 = r["strategy"]["mode"] + (str(r["strategy"]["workers"]) if r["strategy"]["mode"] == "mp" else "")
            measured.setdefault(_key(r), {}).setdefault(r["strategy"]["device"], set()).add(nm0)
    for r in results:
        if not _usable(r) or r["phase"] == "confirmation":
            continue
        k, d = _key(r), r["strategy"]["device"]
        nm = r["strategy"]["mode"] + (str(r["strategy"]["workers"]) if r["strategy"]["mode"] == "mp" else "")
        cur = best.setdefault(k, {}).get(d)
        if cur is None or r["ips"] > cur[0]:
            best[k][d] = (r["ips"], nm)
    out = []
    for k, d in best.items():
        if "cpu" in d and "gpu" in d:
            out.append({"point": k, "best_cpu": d["cpu"][1], "best_cpu_ips": d["cpu"][0], "best_gpu": d["gpu"][1],
                        "best_gpu_ips": d["gpu"][0], "ratio_best_gpu_over_best_cpu": d["gpu"][0] / d["cpu"][0],
                        "population_mode": k.rsplit("/", 1)[-1],
                        "cpu_strategies_measured": sorted(measured.get(k, {}).get("cpu", [])), "gpu_strategies_measured": sorted(measured.get(k, {}).get("gpu", [])),
                        "cpu_multiprocessing_measured": any(x.startswith("mp") for x in measured.get(k, {}).get("cpu", [])),
                        "comparison": "cross-strategy (best available on each device); exploration data"})
    return out


def components(results) -> List[Dict[str, Any]]:
    acc: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in results:
        if r.get("status") != "completed" or r["strategy"]["mode"] != "seq" or r["point"]["mode"] != "hetero" or not r.get("wall_s"):
            continue
        key = (r["strategy"]["device"], r["point"]["tier"])
        a = acc.setdefault(key, {"wall": 0.0, "evals": 0, **{c: 0.0 for c in COMP_ORDER}})
        a["wall"] += r["wall_s"]
        a["evals"] += r["evaluations"]
        for c in COMP_ORDER:
            a[c] += r["comps"].get(c) or 0.0
    rows = []
    for (d, tier), a in sorted(acc.items(), key=lambda kv: (kv[0][0], TIER_ORDER.index(kv[0][1]))):
        rows.append({"device": d, "tier": tier, "wall_s": a["wall"], "evaluations": a["evals"],
                     "share": {c: a[c] / a["wall"] for c in COMP_ORDER}, "seconds": {c: a[c] for c in COMP_ORDER}})
    return rows


def cost_fit(results, st: Settings) -> Dict[str, Any]:
    cm = D.CostModel()
    cm.update([r for r in results if _usable(r) or r.get("strategy", {}).get("device") == "cpu"], st)
    fits = {f"{d}|{t}": v for (d, t), v in cm.fitted.items()}
    be = {}
    for tier in TIER_ORDER:
        g, c = cm.fitted.get(("gpu", tier)), cm.fitted.get(("cpu", tier))
        if g and c:
            s = stats.break_even_steps(g["fixed"], g["step"], c["fixed"], c["step"])
            be[tier] = {"gpu_fixed_s": g["fixed"], "gpu_step_s": g["step"], "cpu_fixed_s": c["fixed"], "cpu_step_s": c["step"],
                        "break_even_steps": s, "within_tested_steps": (s is not None and s <= max(st.steps)),
                        "reason_none": (None if s is not None else
                                        ("GPU per-step cost is not lower than CPU's" if g["step"] >= c["step"] else "GPU fixed cost is not higher: no crossing"))}
    return {"fits": fits, "break_even": be, "kind": "estimated",
            "note": "Two-parameter model per device and tier (fixed seconds per new model + seconds per training step) fitted to "
                    "the measured components; break-even steps is an EXTRAPOLATION when above the tested range."}


def memory_rows(results) -> List[Dict[str, Any]]:
    acc: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for r in results:
        if r.get("status") != "completed" or r.get("peak_additional_gb") is None:
            continue
        s = r["strategy"]
        name = f'{s["device"]}-{s["mode"]}' + (str(s["workers"]) if s["mode"] == "mp" else "")
        acc.setdefault((name, r["point"]["tier"], r["point"]["mode"], r["point"]["population"]), []).append(r)
    rows = []
    for (name, tier, pm, pop), cs in sorted(acc.items()):
        rows.append({"strategy": name, "tier": tier, "population_mode": pm, "population": pop,
                     "peak_additional_gb": max(c["peak_additional_gb"] for c in cs), "retained_gb": _med([c["retained_gb"] for c in cs]),
                     "swap_growth_mb": max(c["swap_growth_mb"] for c in cs), "min_available_gb": min(c["min_avail_gb"] for c in cs if c.get("min_avail_gb") is not None) if any(c.get("min_avail_gb") is not None for c in cs) else None,
                     "n": len(cs)})
    return rows


def memory_slopes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Least-squares MB per additional individual in the population for each (strategy, tier, mode) with >= 3 sizes."""
    out = []
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["strategy"], r["tier"], r["population_mode"]), []).append(r)
    for (name, tier, pm), rs in groups.items():
        rs = sorted(rs, key=lambda r: r["population"])
        if len({r["population"] for r in rs}) >= 3:
            xs, ys = [r["population"] for r in rs], [r["peak_additional_gb"] * 1024 for r in rs]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            den = sum((x - mx) ** 2 for x in xs)
            out.append({"strategy": name, "tier": tier, "population_mode": pm, "mb_per_individual": sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0,
                        "points": len(rs)})
    return out


def incomplete(results, skipped) -> Dict[str, Any]:
    bad = [{"cell_id": r["cell_id"], "status": r["status"], "reason": r.get("reason", "")} for r in results if r.get("status") != "completed"]
    return {"not_completed": bad, "skipped_by_design": skipped, "n_not_completed": len(bad), "n_skipped": len(skipped)}


def analyze(results: List[Dict[str, Any]], st: Settings, skipped: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    pts = seq_points(results, st)
    cross = crossover(pts, st)
    mem = memory_rows(results)
    batched_pts = seq_points(results, st, "batched", ("hetero", "homo"))
    cross["batched"] = crossover_batched(batched_pts)
    an = {"seq_points": pts, "crossover": cross, "batched_points": batched_pts, "strategies": strategy_rows(results, st), "best_available": best_available(results),
          "components": components(results), "cost_model": cost_fit(results, st), "memory": mem, "memory_slopes": memory_slopes(mem),
          "incomplete": incomplete(results, skipped or [])}
    from . import scheduler
    an["scheduler_model"] = scheduler.build_model(results, st)
    an["scheduler_validation"] = scheduler.validate(results, st)
    an["answers"] = build_answers(an, st)
    return an


# ------------------------------------------------------------------------------------------------------ answers
def _fmt_ci(p: Dict[str, Any]) -> str:
    c = p["confirm"]
    if c.get("lo") is not None:
        return f'{p["ratio"]:.2f}x (95% CI {c["lo"]:.2f}-{c["hi"]:.2f}, n={c["n_num"]} GPU / {c["n_den"]} CPU reps)'
    return f'{p["ratio"]:.2f}x (single exploration run, no CI)'


def build_answers(an: Dict[str, Any], st: Settings) -> List[Dict[str, Any]]:
    cr, pts = an["crossover"], an["seq_points"]
    A: List[Dict[str, Any]] = []

    # 1 ------------------------------------------------------------------------------------------------------------
    best = cr["best"]
    if cr["gpu_faster_confirmed"]:
        ps = [p for p in pts if p["key"] in cr["gpu_faster_confirmed"]]
        txt = "YES (confirmed): GPU sequential execution was faster than CPU sequential at " + "; ".join(f'{p["key"]}: {_fmt_ci(p)}' for p in ps[:4]) + "."
        conf = "confirmed"
    elif cr["gpu_faster_explored_only"]:
        txt = ("POSSIBLY: exploration runs show GPU > CPU at " + ", ".join(cr["gpu_faster_explored_only"][:4]) +
               ", but these were not confirmed with independent repetitions, so no claim is made.")
        conf = "unconfirmed"
    elif best:
        txt = (f"NO, not in the tested region. The best GPU/CPU throughput ratio was {_fmt_ci(best)} at {best['key']}; "
               f"CPU was confirmed faster at {len(cr['cpu_faster_confirmed'])} point(s).")
        conf = "confirmed" if best["confirm"].get("lo") is not None else "unconfirmed"
    else:
        txt, conf = "NOT MEASURED: no paired CPU/GPU data were collected (see incomplete configurations).", "none"
    bw = [p for p in an.get("batched_points", []) if p["verdict"] == "gpu_faster"]
    if bw:
        txt = ("YES for the BATCHED strategy (confirmed, identical strategy GPU-batched vs CPU-batched): "
               + "; ".join(f'{p["key"]}: {_fmt_ci(p)}' for p in bw[:4]) + ". For SEQUENTIAL evaluation: " + txt[0].lower() + txt[1:])
        conf = "confirmed"
    ba = [b for b in an["best_available"] if b["ratio_best_gpu_over_best_cpu"] > 1.0]
    if ba:
        txt += (" Cross-strategy (best available strategy on each device, exploration data): GPU ahead at "
                + ", ".join(f'{b["point"]} ({b["best_gpu"]} vs {b["best_cpu"]}, {b["ratio_best_gpu_over_best_cpu"]:.2f}x)' for b in ba[:3]) + ".")
        if any(not b["cpu_multiprocessing_measured"] for b in ba):
            txt += (" CAVEAT: CPU multiprocessing was NOT measured at "
                    + ", ".join(b["point"] for b in ba if not b["cpu_multiprocessing_measured"]) + ", so the best CPU strategy there may be faster than shown.")
    A.append({"q": "Is GPU execution faster for any tested neuroevolution configuration?", "a": txt, "confidence": conf})

    # 2 ------------------------------------------------------------------------------------------------------------
    comp = [p for p in pts if p["ratio"] is not None and p["ratio"] >= COMPETITIVE]
    be = an["cost_model"]["break_even"]
    lines = []
    if not any(p["ratio"] is not None for p in pts):
        lines.append("NOT MEASURED: there is no paired CPU/GPU measurement, so nothing can be said about where the GPU becomes competitive.")
    elif comp:
        lines.append(f"GPU reached >= {COMPETITIVE:.0%} of CPU throughput (a definition of 'competitive', not a measurement) at: " +
                     ", ".join(f'{p["key"]} ({p["ratio"]:.2f}x)' for p in comp[:6]) + ".")
    else:
        lines.append(f"The GPU never reached {COMPETITIVE:.0%} of CPU throughput in the tested region.")
    for e in cr.get("batched", {}).get("confirmed_crossings", [])[:2]:
        lines.append(f"Confirmed crossover of the batched strategy along population size between P={e['between'][0]:g} and P={e['between'][1]:g} ({e['tier']}, S={e['steps']}, {e['population_mode']}).")
    for e in cr.get("batched", {}).get("candidate_crossings", [])[:2]:
        lines.append(f"Candidate (NOT confirmed) batched crossover along population size between P={e['between'][0]:g} and P={e['between'][1]:g} ({e['tier']}, S={e['steps']}, {e['population_mode']}).")
    for e in cr["confirmed_crossings"][:2]:
        lines.append(f"Confirmed crossover along {e['axis']} between {e['between'][0]:g} and {e['between'][1]:g}"
                     + (f" (interpolated ~{e['interpolated_x']:.3g})" if e.get("interpolated_x") else "") + ".")
    for tier, b in be.items():
        if b["break_even_steps"] is not None:
            lines.append(f"ESTIMATED (model extrapolation, not measured) break-even for tier '{tier}': ~{b['break_even_steps']:.0f} training steps per individual"
                         + (" (inside the tested range)." if b["within_tested_steps"] else f" (beyond the tested maximum of {max(st.steps)})."))
        elif b["reason_none"]:
            lines.append(f"Tier '{tier}': no break-even from the fitted model ({b['reason_none']}).")
    A.append({"q": "At what workload sizes does GPU execution become competitive?", "a": " ".join(lines),
              "confidence": "confirmed" if cr["confirmed_crossings"] else ("none" if not any(p["ratio"] is not None for p in pts) else "estimated")})

    # 3 ------------------------------------------------------------------------------------------------------------
    parts = []
    for c in an["components"]:
        s = c["share"]
        fixed = s["construct"] + s["trace"] + s["first_exec"] + s["fitness"]
        parts.append(f'{c["device"]}/{c["tier"]}: per-model fixed work (construct+trace+first execution+fitness) {fixed:.0%}, training {s["train"]:.0%}, '
                     f'GA+cleanup {s["ga"] + s["cleanup"]:.1%}, unattributed {s["unattributed"]:.1%}')
    A.append({"q": "Which components dominate total runtime?", "a": "; ".join(parts) if parts else "NOT MEASURED.", "confidence": "measured"})

    # 4 ------------------------------------------------------------------------------------------------------------
    batched = [r for r in an["strategies"] if r["strategy"].endswith("batched")]
    if batched:
        def rng(sel):
            v = [r["speedup_vs_sequential"] for r in sel if r["speedup_vs_sequential"]]
            return f"{min(v):.2f}-{max(v):.2f}x (n={len(v)})" if v else "n/a"
        hom = [r for r in batched if r["population_mode"] == "homo"]
        het = [r for r in batched if r["population_mode"] == "hetero"]
        cov = _med([r["batched_member_fraction"] for r in het if r["batched_member_fraction"] is not None])
        txt = (f"Batched vs sequential on the SAME device and workload (cross-strategy): homogeneous populations GPU "
               f"{rng([r for r in hom if r['device'] == 'gpu'])}, CPU {rng([r for r in hom if r['device'] == 'cpu'])}; heterogeneous populations "
               f"GPU {rng([r for r in het if r['device'] == 'gpu'])}, CPU {rng([r for r in het if r['device'] == 'cpu'])}"
               + (f"; only {cov:.0%} of heterogeneous individuals fell into a batchable bucket (median)." if cov is not None else "."))
        bp = [p for p in an.get("batched_points", []) if p["ratio"] is not None]
        if bp:
            txt += (" Identical-strategy comparison, GPU-batched vs CPU-batched (GPU/CPU throughput): "
                    + "; ".join(f'{p["key"]}: {_fmt_ci(p)}' for p in bp[:4]) + ".")
        conf = "confirmed" if any(r["evidence"] == "confirmed" for r in batched) else "unconfirmed"
    else:
        txt, conf = "NOT MEASURED in this run (batched strategy not scheduled or not completed).", "none"
    A.append({"q": "Does GPU batching materially improve throughput?", "a": txt, "confidence": conf})

    # 5 ------------------------------------------------------------------------------------------------------------
    mem = an["memory"]
    if mem:
        mx = max(mem, key=lambda m: m["peak_additional_gb"])
        sl = [s for s in an["memory_slopes"] if s["strategy"].endswith("batched")]
        txt = (f"Largest additional memory in any completed configuration: {mx['peak_additional_gb']:.2f} GB ({mx['strategy']}, {mx['tier']}, "
               f"P={mx['population']}, {mx['population_mode']}); maximum swap growth {max(m['swap_growth_mb'] for m in mem):.0f} MB. "
               + ("Batched strategies grow with population (" + ", ".join(f"{s['strategy']}/{s['tier']}/{s['population_mode']}: {s['mb_per_individual']:.1f} MB per individual" for s in sl[:4]) + "); "
                  if sl else "") + "sequential strategies discard each model so their peak depends on model size, not population.")
    else:
        txt = "NOT MEASURED."
    A.append({"q": "How does memory usage scale with population and model complexity?", "a": txt, "confidence": "measured"})

    # 6 ------------------------------------------------------------------------------------------------------------
    n_conf = cr["n_confirmed_points"]
    contra = [p for p in pts if p["explore_ratio"] and p["confirm"].get("ratio") and abs(math.log(p["explore_ratio"] / p["confirm"]["ratio"])) > math.log(1.25)]
    txt = (f"{n_conf} point(s) have confirmation repetitions. Crossover status: {cr['status']}. "
           + (f"Exploration and confirmation disagreed by more than 25% at {', '.join(p['key'] for p in contra[:3])}: exploration single runs are noisier than confirmed ones. " if contra else
              "Confirmation agreed with exploration within 25% wherever both exist. ")
           + ("No crossover is claimed because none was confirmed." if not cr["confirmed_crossings"] else "A crossover is claimed only where adjacent confirmed points changed from CPU-faster to GPU-faster."))
    A.append({"q": "What evidence supports or contradicts the proposed crossover?", "a": txt, "confidence": "confirmed" if n_conf else "none"})

    # 7 ------------------------------------------------------------------------------------------------------------
    g = [c for c in an["components"] if c["device"] == "gpu"]
    fixed_g = {c["tier"]: (c["share"]["construct"] + c["share"]["trace"] + c["share"]["first_exec"]) for c in g}
    txt = ("Observed on this software/hardware combination only: "
           + (f"per-model graph tracing/first execution accounts for {max(fixed_g.values()):.0%} of GPU wall time at its worst tier, "
              "consistent with a per-model compilation/dispatch cost on tensorflow-metal; " if fixed_g else "")
           + "strict-placement kernel gaps are reported in the block table; results are valid only for the recorded TensorFlow / "
             "tensorflow-metal versions and `--grappler` setting. Whether this cost is a Metal-plugin limitation or a property of the M1 Pro GPU "
             "is NOT separable with this experiment.")
    A.append({"q": "Which findings are specific to TensorFlow Metal and this M1 Pro?", "a": txt, "confidence": "hypothesis"})

    # 8 ------------------------------------------------------------------------------------------------------------
    unc = [f"only {st.confirm_reps} independent repetitions per confirmed point (wide intervals)",
           "one machine, one session; day-to-day variance (up to ~6% on throughput in earlier baselines) is not included in the CIs",
           "cold vs warm Metal shader cache can change first-execution cost between sessions",
           "break-even steps are model extrapolations", "synthetic regression task and small dense networks only",
           "paired comparisons use identical genomes/data but GPU reductions are not bit-reproducible"]
    if an["incomplete"]["n_not_completed"] or an["incomplete"]["n_skipped"]:
        unc.append(f'{an["incomplete"]["n_not_completed"]} configuration(s) did not complete and {an["incomplete"]["n_skipped"]} were skipped by the budget/limits')
    A.append({"q": "Which conclusions remain uncertain?", "a": "; ".join(unc) + ".", "confidence": "n/a"})
    return A


def purchase_implications(an: Dict[str, Any], st: Settings) -> List[str]:
    cr = an["crossover"]
    out = []
    if cr["status"] == "none_in_tested_range" or not cr["gpu_faster_confirmed"]:
        out.append("For neuroevolution workloads like those tested (new model per individual, <= 50 steps), the CPU path was at least as fast; "
                   "buying more GPU cores alone is not supported by this evidence for such workloads.")
    else:
        out.append("A region where the GPU wins was found (see Q1). A GPU with more compute only helps if your workloads sit in that region.")
    out.append("The per-model fixed cost (tracing, first execution) is the term that decides the outcome for small networks; this suite cannot "
               "tell whether a different Apple Silicon chip reduces it. Only running `tf_memory_stress_test.py --tests scaling` on that machine can.")
    mp = [r for r in an["strategies"] if r["strategy"].startswith("cpu-mp") and r["speedup_vs_sequential"]]
    if mp:
        best = max(mp, key=lambda r: r["speedup_vs_sequential"])
        out.append(f"CPU multiprocessing reached {best['speedup_vs_sequential']:.2f}x over sequential CPU ({best['strategy']}, efficiency "
                   f"{(best['parallel_efficiency'] or 0):.0%}); if your engine can evaluate individuals in parallel, CPU core count is a relevant purchase dimension.")
    out.append("Memory: see Q5; peak additional memory in these experiments is the measured figure to compare with candidate RAM sizes. No performance "
               "numbers are inferred for untested processors.")
    return out
