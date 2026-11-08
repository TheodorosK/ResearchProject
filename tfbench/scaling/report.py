"""Markdown / CSV / SVG artifacts for the scaling benchmark."""
from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, List, Optional

from .. import svgcharts
from ..report import _f, _table
from .analysis import COMP_ORDER, purchase_implications
from .workload import TIER_ORDER, Point, Settings, point_param_stats

CELL_COLUMNS = ["cell_id", "phase", "rep", "tier", "population", "population_mode", "steps", "generations", "strategy", "device",
                "placement", "status", "reason", "evaluations", "failed_evaluations", "wall_s", "ips", "steps_per_s", "train_steps_per_s",
                "construct_s", "trace_s", "first_exec_s", "train_s", "fitness_s", "teardown_s", "ga_s", "cleanup_s", "unattributed_s",
                "peak_additional_gb", "swap_growth_mb", "retained_gb", "retained_sys_gb", "min_avail_gb", "batched_member_fraction", "padded_ratio_mean",
                "best_val_r2", "signature"]


def cells_csv(cells: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CELL_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for r in cells:
        p, s = r["point"], r["strategy"]
        row = {**r, "tier": p["tier"], "population": p["population"], "population_mode": p["mode"], "steps": p["steps"],
               "generations": p["generations"], "device": s["device"],
               "strategy": f'{s["device"]}-{s["mode"]}' + (str(s["workers"]) if s["mode"] == "mp" else "")}
        for k in ("construct", "trace", "first_exec", "train", "fitness", "teardown", "ga", "cleanup", "unattributed"):
            row[f"{k}_s"] = (r.get("comps") or {}).get(k)
        w.writerow(row)
    return buf.getvalue()


def build_charts(an: Dict[str, Any], st: Settings) -> Dict[str, str]:
    charts: Dict[str, str] = {}
    pts = [p for p in an["seq_points"] if p["ratio"] is not None and p["population"] == st.pilot_pop]
    ser: Dict[str, list] = {}
    for p in pts:
        c = p["confirm"]
        ser.setdefault(p["tier"], []).append((p["steps"], p["ratio"], c.get("lo"), c.get("hi")))
    charts["ratio_vs_steps.svg"] = svgcharts.line_chart(
        f"GPU/CPU throughput ratio vs training steps (P={st.pilot_pop}; whiskers = 95% CI where confirmed)", ser,
        "training steps per individual", "GPU / CPU individuals per second")
    ser2: Dict[str, list] = {}
    for p in pts:
        x = point_param_stats(Point(p["tier"], p["population"], p["steps"]), st)["mean"]
        c = p["confirm"]
        ser2.setdefault(f'S={p["steps"]}', []).append((x, p["ratio"], c.get("lo"), c.get("hi")))
    charts["ratio_vs_model_size.svg"] = svgcharts.line_chart("GPU/CPU throughput ratio vs network size", ser2, "mean parameters per network",
                                                             "GPU / CPU individuals per second")
    steps = sorted({p["steps"] for p in pts})
    tiers = [t for t in TIER_ORDER if t in {p["tier"] for p in pts}]
    grid = [[next((p["ratio"] for p in pts if p["tier"] == t and p["steps"] == s), None) for s in steps] for t in tiers]
    charts["heatmap_ratio.svg"] = svgcharts.heatmap(f"GPU/CPU ratio, tier x steps (P={st.pilot_pop})", tiers, [f"S={s}" for s in steps], grid)
    comp = an["components"]
    groups = [f'{c["device"]}/{c["tier"]}' for c in comp]
    charts["time_components.svg"] = svgcharts.stacked_bars(
        "Where the time goes (sequential strategy)", groups,
        {"per-model setup (construct+trace)": [c["share"]["construct"] + c["share"]["trace"] for c in comp],
         "first execution": [c["share"]["first_exec"] for c in comp], "training": [c["share"]["train"] for c in comp],
         "fitness": [c["share"]["fitness"] for c in comp],
         "GA+cleanup+other": [c["share"]["ga"] + c["share"]["cleanup"] + (c["share"]["unattributed"] or 0) for c in comp]})
    return charts


def markdown(job: Dict[str, Any], chart_dir: str = "charts") -> str:
    d = job["data"]
    an, st = d["analysis"], Settings.from_dict(d["settings"])
    ctl, design = d["controller"], d.get("design_summary", {})
    L: List[str] = []
    A = L.append
    if d.get("synthetic"):
        A("> **SYNTHETIC ILLUSTRATION - not a measurement.** Generated from a known ground-truth cost model to show the report layout and to test the analysis; "
          "it says nothing about any real machine.\n")
    A("## 11. Neuroevolution CPU/GPU scaling and crossover\n")
    A(f"**Status:** {job['status']}{(' - ' + job['reason']) if job.get('reason') else ''}. "
      f"Elapsed {ctl.get('elapsed_s', 0) / 60:.1f} min of a {st.max_total_s / 60:.0f} min budget (per-configuration limit {st.max_cell_s / 60:.0f} min).\n")
    cells = ctl["cells"]
    done = [c for c in cells if c.get("status") == "completed"]
    A("### 11.1 Design and coverage\n")
    A(_table(["Item", "Value"], [
        ["Exhaustive grid (tiers x populations x steps)", f'{design.get("full_cartesian_points", "n/a")} workload points (x strategies x repetitions - NOT run)'],
        ["Cell-runs executed (completed / not completed)", f"{len(done)} / {len(cells) - len(done)}"],
        ["Skipped by budget or limits", len(ctl["skipped"])],
        ["Pilot / expansion / confirmation completed", " / ".join(str(sum(1 for c in done if c["phase"] == ph)) for ph in ("pilot", "expansion", "confirmation"))],
        ["Candidates carried into expansion", ", ".join(ctl["phase_log"].get("candidates", [])) or "none"],
        ["Worker processes started", len(ctl["blocks"])],
        ["Workload", f"batch {st.batch}, input dim {st.in_dim}, G={st.generations} generations, activations {', '.join(st.activations)}, seed {st.seed}"]]))
    A("Exploration (pilot, expansion) uses single runs and only SELECTS what to confirm; verdicts below come from independent "
      f"repetitions (`confirmation`, {st.confirm_reps} fresh processes, device order alternated).\n")
    A("### 11.2 Answers to the research questions\n")
    for i, a in enumerate(an["answers"], 1):
        A(f"**{i}. {a['q']}** *(evidence: {a['confidence']})*\n\n{a['a']}\n")
    A("### 11.3 Crossover analysis (identical strategy: CPU sequential vs GPU sequential)\n")
    cr = an["crossover"]
    A(f"**Crossover status: `{cr['status']}`** - confirmed points: {cr['n_confirmed_points']} of {cr['n_points']}.\n")
    rows = []
    for p in an["seq_points"]:
        c = p["confirm"]
        rows.append([p["tier"], p["population"], p["steps"], p["explore_ratio"], (f'{c["ratio"]:.3g} [{c["lo"]:.3g}, {c["hi"]:.3g}]' if c.get("lo") is not None else "n/a"),
                     f'{c["n_num"]}/{c["n_den"]}', (c["cpu"].get("cv")), (c["gpu"].get("cv")), p["verdict"]])
    A(_table(["Tier", "P", "Steps", "Exploration GPU/CPU", "Confirmed GPU/CPU [95% CI]", "Reps G/C", "CPU CV", "GPU CV", "Verdict"], rows))
    for e in cr["confirmed_crossings"]:
        A(f"- **Confirmed** crossover along {e['axis']} between {e['between'][0]:g} and {e['between'][1]:g}"
          + (f" (interpolated ~{e['interpolated_x']:.3g})" if e.get("interpolated_x") else "") + ".")
    for e in cr["candidate_crossings"]:
        A(f"- *Candidate, NOT confirmed*: ratio crosses 1 along {e['axis']} between {e['between'][0]:g} and {e['between'][1]:g}.")
    A("")
    bp = [p for p in an.get("batched_points", []) if p["ratio"] is not None]
    if bp:
        A("**Batched strategy, CPU vs GPU (identical strategy, same workload):**\n")
        A(_table(["Point", "Exploration GPU/CPU", "Confirmed [95% CI]", "Verdict"],
                 [[p["key"], p["explore_ratio"], (f'{p["confirm"]["ratio"]:.3g} [{p["confirm"]["lo"]:.3g}, {p["confirm"]["hi"]:.3g}]' if p["confirm"].get("lo") is not None else "n/a"), p["verdict"]] for p in bp]))
    for e in cr.get("batched", {}).get("confirmed_crossings", []):
        A(f"- **Confirmed** batched crossover along population size between P={e['between'][0]:g} and P={e['between'][1]:g} ({e['tier']}, S={e['steps']}, {e['population_mode']})"
          + (f"; log-linear interpolation ~P={e['interpolated_x']:.2g} (rough)." if e.get("interpolated_x") else "."))
    for e in cr.get("batched", {}).get("candidate_crossings", []):
        A(f"- *Candidate, NOT confirmed*: batched GPU/CPU ratio crosses 1 between P={e['between'][0]:g} and P={e['between'][1]:g} ({e['tier']}, S={e['steps']}, {e['population_mode']}).")
    for name in ("ratio_vs_steps.svg", "ratio_vs_model_size.svg", "heatmap_ratio.svg"):
        A(f"![{name}]({chart_dir}/{name})\n")
    A("### 11.4 Timing decomposition (sequential strategy, share of wall time)\n")
    A(_table(["Device", "Tier", "Wall (s)", "Evals", "Construct", "Trace", "First exec", "Train", "Fitness", "Teardown", "GA", "Cleanup", "Unattributed"],
             [[c["device"], c["tier"], c["wall_s"], c["evaluations"]] + [f'{c["share"][k]:.1%}' for k in COMP_ORDER] for c in an["components"]]))
    A("Components are measured per individual with host read-back barriers; *First exec* includes graph optimisation, kernel/shader compilation and training step #1.\n")
    A(f"![time_components.svg]({chart_dir}/time_components.svg)\n")
    A("### 11.5 Strategies: batching and CPU multiprocessing (different strategy, same workload and device)\n")
    A(_table(["Point", "Strategy", "Pop. mode", "Speed-up vs sequential", "95% CI", "Evidence", "Batchable members", "Padding x", "MP efficiency", "Pool start-up (s)"],
             [[r["point"], r["strategy"], r["population_mode"], r["speedup_vs_sequential"], (f'[{r["ci_lo"]:.3g}, {r["ci_hi"]:.3g}]' if r["ci_lo"] else "n/a"), r["evidence"],
               r["batched_member_fraction"], r["padded_ratio"], r["parallel_efficiency"], r["pool_startup_s"]] for r in an["strategies"]]))
    if an["best_available"]:
        A("**Best available strategy per device (cross-strategy, exploration data):**\n")
        A(_table(["Point", "Pop. mode", "Best CPU", "CPU ind/s", "Best GPU", "GPU ind/s", "GPU/CPU", "CPU multiprocessing measured"],
                 [[b["point"], b.get("population_mode"), b["best_cpu"], b["best_cpu_ips"], b["best_gpu"], b["best_gpu_ips"], b["ratio_best_gpu_over_best_cpu"], b.get("cpu_multiprocessing_measured")] for b in an["best_available"]]))
    A("### 11.6 Memory\n")
    A(_table(["Strategy", "Tier", "Pop. mode", "P", "Peak additional (GB)", "Retained after cleanup (GB)", "Swap growth (MB)", "n"],
             [[m["strategy"], m["tier"], m["population_mode"], m["population"], m["peak_additional_gb"], m["retained_gb"], m["swap_growth_mb"], m["n"]] for m in an["memory"]]))
    A("Retained = process RSS after the cell's final `gc.collect()` minus RSS before the cell; allocator caching is NOT distinguished from retention here "
      "(use `memdiag` for that).\n")
    A("### 11.7 Cost model and ESTIMATED break-even (not measured)\n")
    A(_table(["Tier", "GPU fixed/model (s)", "GPU s/step", "CPU fixed/model (s)", "CPU s/step", "Break-even steps (estimated)", "Inside tested range"],
             [[t, b["gpu_fixed_s"], b["gpu_step_s"], b["cpu_fixed_s"], b["cpu_step_s"], (f'{b["break_even_steps"]:.0f}' if b["break_even_steps"] else (b["reason_none"] or "n/a")), b["within_tested_steps"]]
              for t, b in an["cost_model"]["break_even"].items()]))
    A(an["cost_model"]["note"] + "\n")
    A("### 11.8 Execution-policy advisor (recommendation mode only)\n")
    sv = an["scheduler_validation"]
    A(_table(["Device", "Held-out MAPE", "Cells", "Per held-out tier"],
             [[dev, f'{v["mape"]:.1%}', v["n_cells"], ", ".join(f"{k}: {x:.0%}" for k, x in v["per_held_out_tier"].items())] for dev, v in sv["per_device"].items()]))
    A(f"Advisor reliable (held-out error <= 30% on both devices): **{sv['reliable']}**. {sv['note']} "
      "Use `python tools/recommend_device.py <report.json> --params ... --layers ... --steps ... --population ...`; it prints advice only and never switches devices.\n")
    A("### 11.9 Incomplete, aborted, skipped and unsupported configurations\n")
    inc = an["incomplete"]
    A(_table(["Cell", "Status", "Reason"], [[x["cell_id"], x["status"], x["reason"][:140]] for x in inc["not_completed"]]) if inc["not_completed"] else "No configuration ended aborted/failed/unsupported.\n")
    if inc["skipped_by_design"]:
        A("Skipped before running (budget, memory estimate, pruning):\n")
        A(_table(["Cell", "Reason"], [[x["cell_id"], x["reason"][:150]] for x in inc["skipped_by_design"][:60]]))
        if len(inc["skipped_by_design"]) > 60:
            A(f"... and {len(inc['skipped_by_design']) - 60} more (see report.json).\n")
    if ctl.get("unusable_strategies"):
        A("Unavailable strategies: " + "; ".join(f"`{k}` - {v}" for k, v in ctl["unusable_strategies"].items()) + "\n")
    A("An aborted, failed, skipped or unsupported configuration is never counted as a performance measurement.\n")
    A("### 11.10 Hardware-purchasing implications and limitations\n")
    for s in purchase_implications(an, st):
        A(f"- {s}")
    A("- Valid only for the recorded TensorFlow / tensorflow-metal versions, `--grappler` setting, `SimpleAdam` optimizer and synthetic regression task.")
    A("- Exploration runs are single measurements; CIs use only confirmation repetitions (small n -> wide intervals).")
    A("- Multiprocessing uses CPU only, one thread per worker; GPU multiprocessing is deliberately not supported (shared Metal device and unified memory).")
    A("- Batching of heterogeneous populations uses bucketing by (depth, activations) with zero-padding of widths; its coverage and padding overhead are reported, not hidden.")
    A("- No performance is inferred for any Apple Silicon processor that was not benchmarked.\n")
    return "\n".join(L)


def write_artifacts(out_dir: str, job: Dict[str, Any]) -> Dict[str, str]:
    d = job["data"]
    base = os.path.join(out_dir, "scaling")
    os.makedirs(os.path.join(base, "charts"), exist_ok=True)
    st = Settings.from_dict(d["settings"])
    paths = {}
    for name, svg in build_charts(d["analysis"], st).items():
        p = os.path.join(base, "charts", name)
        with open(p, "w") as fh:
            fh.write(svg)
        paths[name] = p
    with open(os.path.join(base, "cells.csv"), "w", newline="") as fh:
        fh.write(cells_csv(d["controller"]["cells"]))
    with open(os.path.join(base, "scaling_summary.md"), "w") as fh:
        fh.write(markdown(job, "charts"))
    import json
    from ..common import sanitize
    with open(os.path.join(base, "crossover.json"), "w") as fh:
        json.dump(sanitize({"crossover": d["analysis"]["crossover"], "answers": d["analysis"]["answers"],
                            "break_even_estimated": d["analysis"]["cost_model"]["break_even"]}), fh, indent=2, allow_nan=False)
    if d["analysis"].get("capacity"):
        from . import capacity
        with open(os.path.join(base, "capacity_summary.md"), "w") as fh:
            fh.write(capacity.markdown(d["analysis"]["capacity"], st, job["status"], d["controller"]["elapsed_s"], d["controller"]["skipped"]))
        paths["capacity"] = os.path.join(base, "capacity_summary.md")
    paths.update(cells_csv=os.path.join(base, "cells.csv"), summary=os.path.join(base, "scaling_summary.md"), crossover=os.path.join(base, "crossover.json"))
    return paths
