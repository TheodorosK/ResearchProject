"""Build JSON / CSV / Markdown reports from per-job result files."""
from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, Iterable, List, Optional

from . import SCHEMA_VERSION, __version__
from .common import atomic_write_json, sanitize

LIMITATIONS = [
    "Single machine, single session: the coefficient of variation (CV) captures repeat-to-repeat noise only; "
    "session-to-session variance (background load, temperature, OS state) is unknown. Repeat on separate days "
    "before trusting small differences.",
    "System-memory figures come from psutil (macOS 'available' memory). They are coarse: they include file cache, "
    "macOS compresses memory, and values can lag or go negative. Peaks are sampled every ~0.5 s, so shorter spikes "
    "are missed.",
    "Process RSS excludes Metal/unified-memory buffers; GPU memory use is therefore visible only as a drop in "
    "system-available memory. tensorflow-metal does not implement tf.config.experimental.get_memory_info.",
    "GPU utilisation, GPU clocks and GPU temperature are not exposed through TensorFlow; `powermetrics` needs "
    "sudo and is not used. pmset thermal fields are coarse and may be absent - absence is NOT evidence of no throttling.",
    "TensorFlow offers no public API for the final per-op placement of a graph function. Device placement is "
    "verified from a probe op, every model variable, and (with soft placement OFF) from the fact that an op "
    "without a GPU kernel raises instead of moving. If placement is 'soft', some ops may have run on the CPU.",
    "Resource limits are enforced by an in-process watchdog (cooperative: checked between steps/individuals) and "
    "by an external supervisor that SIGTERM/SIGKILLs a worker that stays over a limit. A native call blocked "
    "inside Metal cannot be interrupted from Python; the supervisor is the backstop.",
    "GPU floating-point reductions are not bit-reproducible: seeds fix data, initialisation and the GA lineage, "
    "but numeric results can differ slightly between runs and between CPU and GPU.",
    "Workloads are synthetic: a dense MLP, a small convnet and small evolved MLPs. They do not represent "
    "transformers, large convolutional models or data-pipeline-bound training. Neuroevolution models are tiny, "
    "so that benchmark is dominated by per-model Keras/TensorFlow overhead and may favour the CPU.",
    "FLOP/s and pre-run memory figures are analytical ESTIMATES, not measurements.",
    "No performance claim is made about other Apple Silicon chips or memory sizes. Comparisons require actually "
    "running this suite (same TensorFlow/tensorflow-metal versions, seeds and settings) on the other machine.",
]

KIND_LEGEND = {
    "measured": "read directly from a clock, OS counter or TensorFlow",
    "calculated": "derived arithmetically from measured values (median, ratio, rate, slope)",
    "estimated": "analytical model or heuristic; NOT observed",
}


# ------------------------------------------------------------------- helpers
def _ms(job: Dict[str, Any], name: str, **ctx) -> List[Dict[str, Any]]:
    return [m for m in job.get("measurements", [])
            if m["name"] == name and all(m.get("context", {}).get(k) == v for k, v in ctx.items())]


def val(job: Dict[str, Any], name: str, default=None, **ctx):
    r = _ms(job, name, **ctx)
    return r[-1]["value"] if r else default


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if x != x:
            return "n/a"
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        return f"{x:.{nd}g}" if abs(x) < 1 else f"{x:.{max(nd - 1, 1)}f}"
    return str(x)


def _table(headers: List[str], rows: Iterable[List[Any]]) -> str:
    rows = list(rows)
    if not rows:
        return "_no data_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_f(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _jobs(jobs: List[Dict[str, Any]], test: str) -> List[Dict[str, Any]]:
    return [j for j in jobs if j.get("test") == test]


# ------------------------------------------------------------------- derived
def derive(jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-job metrics. All are 'calculated' and only produced when BOTH sides completed."""
    d: Dict[str, Any] = {"cpu_vs_gpu_performance": [], "cpu_vs_gpu_neuroevolution": None}
    perf = [j for j in _jobs(jobs, "performance") if j.get("status") == "completed"]
    for wl in sorted({j.get("workload") for j in perf}):
        cpu = next((j for j in perf if j["workload"] == wl and j["device"] == "cpu"), None)
        gpu = next((j for j in perf if j["workload"] == wl and j["device"] == "gpu"), None)
        if not (cpu and gpu):
            continue
        c_ms, g_ms = val(cpu, "steady_step_time_ms"), val(gpu, "steady_step_time_ms")
        lc, lg = val(cpu, "loss_first_step"), val(gpu, "loss_first_step")
        rel = abs(lc - lg) / max(abs(lc), abs(lg), 1e-12) if lc is not None and lg is not None else None
        fc, fg = val(cpu, "loss_final"), val(gpu, "loss_final")
        frel = abs(fc - fg) / max(abs(fc), abs(fg), 1e-12) if fc is not None and fg is not None else None
        d["cpu_vs_gpu_performance"].append({
            "final_loss_cpu": fc, "final_loss_gpu": fg, "final_loss_rel_diff": frel,
            "training_trajectories_consistent": (frel is not None and frel < 0.05),
            "workload": wl, "kind": "calculated",
            "cpu_step_ms": c_ms, "gpu_step_ms": g_ms,
            "gpu_speedup_x": (c_ms / g_ms) if c_ms and g_ms else None,
            "cpu_samples_per_s": val(cpu, "steady_samples_per_s"),
            "gpu_samples_per_s": val(gpu, "steady_samples_per_s"),
            "gpu_placement_mode": gpu.get("placement"),
            "first_step_loss_rel_diff": rel,
            "workloads_comparable": (rel is not None and rel < 1e-2),
        })
    ne = [j for j in _jobs(jobs, "neuroevolution") if j.get("status") == "completed"]
    c = next((j for j in ne if j["device"] == "cpu"), None)
    g = next((j for j in ne if j["device"] == "gpu"), None)
    if c and g:
        a, b = val(c, "overall_individuals_per_s"), val(g, "overall_individuals_per_s")
        tc = (c.get("data", {}).get("reference_check") or {}).get("loss_trace") or []
        tg = (g.get("data", {}).get("reference_check") or {}).get("loss_trace") or []
        first_div = next((i for i, (x, y) in enumerate(zip(tc, tg))
                          if abs(x - y) / max(abs(x), abs(y), 1e-12) > 1e-3), None)
        rc, rg = val(c, "reference_val_r2"), val(g, "reference_val_r2")
        diff = abs(rc - rg) if rc is not None and rg is not None else None
        d["cpu_vs_gpu_neuroevolution"] = {
            "kind": "calculated", "cpu_ind_per_s": a, "gpu_ind_per_s": b,
            "gpu_vs_cpu_throughput_x": (b / a) if a and b else None,
            "reference_val_r2_cpu": rc, "reference_val_r2_gpu": rg, "reference_val_r2_abs_diff": diff,
            "numerically_consistent": (diff < 0.05) if diff is not None else None,
            "reference_loss_trace_cpu": tc, "reference_loss_trace_gpu": tg,
            "first_divergent_step_rel_1e-3": first_div,
            "gpu_placement_mode": g.get("placement")}
    return d


# --------------------------------------------------------------------- report
def build_report(meta: Dict[str, Any], jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    incomplete = [{"job": j["job"], "status": j["status"], "reason": j.get("reason", "")}
                  for j in jobs if j.get("status") != "completed"]
    notes: List[str] = []
    for j in jobs:
        for n in j.get("notes", []):
            if n not in notes:
                notes.append(n)
    return {
        "schema_version": SCHEMA_VERSION, "suite_version": __version__,
        "kind_legend": KIND_LEGEND, **meta,
        "jobs": jobs, "derived": derive(jobs), "incomplete": incomplete,
        "notes": notes, "limitations": LIMITATIONS,
    }


CSV_COLUMNS = ["job", "test", "device", "workload", "placement", "name", "value", "unit", "kind",
               "repeat", "generation", "cycle", "policy"]


def render_csv(report: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for j in report["jobs"]:
        for m in j.get("measurements", []):
            ctx = m.get("context", {})
            w.writerow({"job": j["job"], "test": j.get("test"), "device": ctx.get("device", j.get("device")),
                        "workload": ctx.get("workload", j.get("workload")),
                        "placement": ctx.get("placement", j.get("placement")),
                        "name": m["name"], "value": sanitize(m["value"]), "unit": m["unit"], "kind": m["kind"],
                        "repeat": ctx.get("repeat"), "generation": ctx.get("generation"),
                        "cycle": ctx.get("cycle"), "policy": ctx.get("policy")})
    for row in report["derived"]["cpu_vs_gpu_performance"]:
        for k, v in row.items():
            if k not in ("workload", "kind", "gpu_placement_mode"):
                w.writerow({"job": "derived", "test": "performance", "workload": row["workload"],
                            "name": k, "value": sanitize(v), "kind": "calculated"})
    return buf.getvalue()


def render_markdown(r: Dict[str, Any]) -> str:
    L: List[str] = []
    A = L.append
    jobs = r["jobs"]
    hw, sw = r.get("hardware", {}), r.get("software", {})
    A(f"# TensorFlow benchmark report\n")
    A(f"*Run `{r.get('run_id')}` - started {r.get('started_utc')} - finished {r.get('finished_utc')} - "
      f"suite v{r.get('suite_version')}*\n")
    A(f"**Command:** `{r.get('command')}`\n")
    overall = "COMPLETE" if not r["incomplete"] else f"INCOMPLETE ({len(r['incomplete'])} job(s) did not complete)"
    A(f"**Overall status:** {overall}\n")
    A("**How to read numbers:** " + "; ".join(f"**{k}** = {v}" for k, v in KIND_LEGEND.items()) + ".\n")

    A("## 1. Hardware and software configuration\n")
    A(_table(["Item", "Value"], [
        ["Machine", hw.get("model_name") or hw.get("model_identifier") or hw.get("platform")],
        ["Chip", hw.get("chip") or hw.get("machine")],
        ["Cores (P/E, logical)", f"{hw.get('performance_cores')}/{hw.get('efficiency_cores')}, {hw.get('cpu_logical')}"],
        ["GPU cores", hw.get("gpu_cores")],
        ["Unified memory (GB)", hw.get("ram_total_gb")],
        ["OS", hw.get("platform")],
        ["Power source", (r.get("power_source") or {}).get("source")],
        ["Low Power Mode", (r.get("low_power_mode") or {}).get("enabled")],
        ["Python", sw.get("python")],
        ["TensorFlow / Keras", f"{sw.get('tensorflow') or sw.get('tensorflow-macos') or sw.get('tensorflow-cpu')} / {sw.get('keras')}"],
        ["tensorflow-metal", sw.get("tensorflow-metal")],
        ["numpy / psutil", f"{sw.get('numpy')} / {sw.get('psutil')}"],
        ["Seed", str(r.get("config", {}).get("seed"))],
    ]))
    lim = r.get("limits", {})
    A(f"**Safety limits:** extra memory <= {lim.get('max_extra_gb')} GB, available >= {lim.get('min_avail_gb')} GB, "
      f"system use < {lim.get('max_sys_pct')}%, swap growth < {lim.get('max_swap_mb')} MB, "
      f"runtime <= {(lim.get('max_runtime_s') or 0) / 60:.0f} min per benchmark.\n")

    A("## 2. TensorFlow and device configuration\n")
    rows = []
    for j in _jobs(jobs, "performance") + _jobs(jobs, "neuroevolution"):
        ev = j.get("data", {}).get("device_evidence", {})
        tfi = j.get("data", {}).get("tf", {})
        rows.append([j["job"], j.get("status"), j.get("placement"), tfi.get("soft_device_placement"),
                     j.get("data", {}).get("execution_mode",
                                           "graph (Keras model.fit)" if j.get("test") == "neuroevolution" else "n/a"),
                     ev.get("verified"), ev.get("probe_op_device"), str(ev.get("variable_device_counts")),
                     ev.get("optimizer_class")])
    A(_table(["Job", "Status", "Placement", "Soft placement", "Execution", "Placement verified", "Probe op device",
              "Variable devices", "Optimizer"], rows))
    for j in jobs:
        gp = j.get("placement_strict_failure")
        if gp:
            A(f"> **Unsupported GPU op / placement failure** in `{j['job']}` (strict mode): `{gp}`\n")

    A("## 3. CPU and GPU performance (steady state)\n")
    rows = []
    for j in _jobs(jobs, "performance"):
        rows.append([j.get("workload"), j.get("device"), j.get("status"),
                     val(j, "steady_steps_per_s"), val(j, "steady_samples_per_s"),
                     val(j, "steady_step_time_ms"), val(j, "steady_run_s_median"), val(j, "steady_run_s_cv"),
                     val(j, "steady_run_s_min"), val(j, "steady_run_s_max")])
    A(_table(["Workload", "Device", "Status", "Steps/s", "Samples/s", "ms/step", "Median run (s)", "CV",
              "Min run (s)", "Max run (s)"], rows))
    A("_Steps/s, samples/s, ms/step, median and CV are **calculated** from the measured per-repeat times._\n")
    A("**CPU vs GPU (calculated):**\n")
    A(_table(["Workload", "CPU ms/step", "GPU ms/step", "GPU speed-up", "GPU placement", "First-step loss rel. diff",
              "Comparable", "Final-loss rel. diff", "Trajectories agree"],
             [[x["workload"], x["cpu_step_ms"], x["gpu_step_ms"], x["gpu_speedup_x"], x["gpu_placement_mode"],
               x["first_step_loss_rel_diff"], x["workloads_comparable"], x["final_loss_rel_diff"],
               x["training_trajectories_consistent"]]
              for x in r["derived"]["cpu_vs_gpu_performance"]]))
    A("**Phase breakdown (measured, seconds) - never merged with steady-state time:**\n")
    A(_table(["Job", "Data gen", "Data transfer", "Model build", "Optimizer init", "Trace", "First execution",
              "Warm-up total", "Compile overhead (estimated)"],
             [[j["job"], val(j, "data_generation_s"), val(j, "data_transfer_s"), val(j, "model_build_s"),
               val(j, "optimizer_init_s"), val(j, "trace_s"), val(j, "first_execution_s"),
               val(j, "warmup_total_s"), val(j, "compile_overhead_estimate_s")]
              for j in _jobs(jobs, "performance")]))
    A("**Thermal / stability indicators:**\n")
    A(_table(["Job", "pmset throttling indicated", "Drift last-vs-first third (x)"],
             [[j["job"], {True: "YES (indicated)", False: "no limit reported", None: "unavailable"}[
                 val(j, "thermal_throttling_indicated_by_pmset")],
               val(j, "steady_drift_ratio_last_vs_first_third")] for j in _jobs(jobs, "performance")]))
    A("_`None`/`n/a` means the telemetry was unavailable, which does **not** mean no throttling occurred._\n")

    A("## 4. Baseline and peak memory\n")
    rows = []
    for j in jobs:
        peak = val(j, "peak_extra_system_gb_sampled")
        if peak is None:
            continue
        rows.append([j["job"], val(j, "peak_rss_gb"), peak, val(j, "min_available_gb_sampled"),
                     val(j, "max_system_pct_sampled")])
    A(_table(["Job", "Peak RSS (GB)", "Peak extra system memory (GB)", "Min available (GB)", "Max system use (%)"], rows))
    A("_Peak extra memory = drop in system-available memory vs. the job's own baseline (captures unified/GPU "
      "memory that RSS misses). All **measured** (sampled ~0.5 s)._\n")

    A("## 5. Swap activity\n")
    A(_table(["Job", "Swap growth during steady state (MB)", "Page-outs during steady state (MB)"],
             [[j["job"], val(j, "swap_growth_during_steady_mb"), val(j, "swap_pageouts_during_steady_mb")]
              for j in _jobs(jobs, "performance")]))
    ne_swap = []
    for j in _jobs(jobs, "neuroevolution"):
        gens = j.get("data", {}).get("generations", [])
        if gens:
            ne_swap.append([j["job"], max(g["swap_growth_since_start_mb"] for g in gens)])
    A(_table(["Neuroevolution job", "Max swap growth since start (MB)"], ne_swap))

    A("## 6. Neuroevolution throughput\n")
    rows = []
    for j in _jobs(jobs, "neuroevolution"):
        b = j.get("data", {}).get("best_individual", {})
        rows.append([j["job"], j.get("status"), val(j, "total_evaluations"), val(j, "failed_evaluations"),
                     val(j, "overall_individuals_per_s"), val(j, "total_evolution_s"),
                     val(j, "val_r2_improvement_first_to_last_generation"), b.get("val_r2"), b.get("test_r2"),
                     b.get("architecture")])
    A(_table(["Job", "Status", "Evaluations", "Failed", "Individuals/s", "Total (s)", "Val R2 gain (gen0->last)",
              "Best val R2", "Best test R2 (final only)", "Best architecture"], rows))
    cmp_ = r["derived"].get("cpu_vs_gpu_neuroevolution")
    if cmp_:
        A(f"**GPU vs CPU neuroevolution throughput (calculated):** {_f(cmp_['gpu_vs_cpu_throughput_x'])}x "
          f"(GPU placement: {cmp_.get('gpu_placement_mode')})\n")
        ok = cmp_.get("numerically_consistent")
        if cmp_.get("reference_loss_trace_cpu") and cmp_.get("reference_loss_trace_gpu"):
            A("**Reference loss trace, first optimiser steps (measured; step 0 = forward pass at identical initial weights):**\n")
            A(_table(["Step", "CPU loss", "GPU loss", "Rel. diff"],
                     [[i, x, y, abs(x - y) / max(abs(x), abs(y), 1e-12)] for i, (x, y) in
                      enumerate(zip(cmp_["reference_loss_trace_cpu"], cmp_["reference_loss_trace_gpu"]))]))
            A(f"First step where CPU and GPU differ by > 0.1%: {cmp_.get('first_divergent_step_rel_1e-3')} "
              "(0 = forward pass differs; later = gradient/optimiser differs; None = no divergence).\n")
        A(f"**Numerical agreement (fixed reference genome, calculated):** CPU val R2 {_f(cmp_['reference_val_r2_cpu'])}, "
          f"GPU val R2 {_f(cmp_['reference_val_r2_gpu'])}, |diff| {_f(cmp_['reference_val_r2_abs_diff'])} -> "
          + ("consistent" if ok else ("**NOT consistent - GPU fitness values must not be compared with CPU ones**"
                                       if ok is False else "unknown")) + "\n")
    for j in _jobs(jobs, "neuroevolution"):
        pp = j.get("data", {}).get("parallel_probe")
        if pp:
            A(f"**Parallel probe `{j['job']}`** (measured): {pp['workers']} workers on {pp['individuals']} individuals - "
              f"sequential {_f(pp['sequential_s'])} s, parallel {_f(pp['parallel_s'])} s, "
              f"speed-up {_f(pp['speedup'])}x (calculated).\n")

    A("## 7. Time per generation\n")
    for j in _jobs(jobs, "neuroevolution"):
        gens = j.get("data", {}).get("generations", [])
        A(f"**{j['job']}**\n")
        A(_table(["Gen", "Evaluated", "Failed", "Wall (s)", "Ind/s", "Mean fit time (s)", "Best val R2",
                  "Mean val R2", "CPU cores busy", "System CPU %"],
                 [[g["generation"], g["evaluated"], g["failed"], g["wall_s"], g["individuals_per_s"],
                   (g["train_s"] or {}).get("mean"), g["best_val_r2"], g["mean_val_r2"],
                   g["process_cores_busy"], g["system_cpu_pct"]] for g in gens]))

    A("## 8. Memory growth per generation / diagnostic\n")
    for j in _jobs(jobs, "neuroevolution"):
        gens = j.get("data", {}).get("generations", [])
        A(f"**{j['job']}**\n")
        A(_table(["Gen", "RSS before cleanup (GB)", "RSS after (GB)", "Extra system after (GB)",
                  "Peak extra (GB)", "Available after (GB)", "Swap since start (MB)", "Cleanup (s)"],
                 [[g["generation"], g["rss_gb_before_cleanup"], g["rss_gb_after_cleanup"],
                   g["extra_system_gb_after_cleanup"], g["peak_extra_system_gb"], g["available_gb_after_cleanup"],
                   g["swap_growth_since_start_mb"], g["cleanup_s"]] for g in gens]))
        A(f"Growth per generation (calculated): system {_f(val(j, 'system_memory_growth_per_generation_mb'))} MB, "
          f"RSS {_f(val(j, 'rss_growth_per_generation_mb'))} MB. "
          "Rising RSS alone is not a leak: allocator caching also grows then plateaus.\n")
    md = _jobs(jobs, "memdiag")
    if md:
        A("**Memory-growth diagnostic (`memdiag`, each policy in its own process; verdicts are heuristic):**\n")
        rows = []
        for j in md:
            c = j.get("data", {}).get("classification", {})
            rows.append([j["job"], j.get("status"), (c.get("rss") or {}).get("verdict"),
                         (c.get("rss") or {}).get("slope_mb_per_cycle"),
                         (c.get("system_extra") or {}).get("verdict"),
                         (c.get("system_extra") or {}).get("slope_mb_per_cycle"),
                         (c.get("rss") or {}).get("python_slope_mb_per_cycle")])
        A(_table(["Job", "Status", "RSS verdict", "RSS MB/cycle", "System verdict", "System MB/cycle",
                  "Python MB/cycle"], rows))
        A("_`bounded`/`saturating` = consistent with allocator caching; `python_retention` = references still held; "
          "`continued_native_growth` = potential native leak, needs a longer run to confirm._\n")

    A("## 9. Incomplete or aborted tests\n")
    A(_table(["Job", "Status", "Reason"], [[i["job"], i["status"], i["reason"]] for i in r["incomplete"]])
      if r["incomplete"] else "All requested jobs completed.\n")
    for j in jobs:
        legacy = {k: val(j, k) for k in ("train_largest_batch_completed", "train_peak_used_gb",
                                         "alloc_requested_gb", "leak_verdict") if val(j, k) is not None}
        if legacy:
            A(f"Legacy `{j['job']}`: {legacy}\n")

    A("## 10. Measurement limitations and notes\n")
    for n in r.get("notes", []):
        A(f"- {n}")
    A("")
    for n in r["limitations"]:
        A(f"- {n}")
    A("")
    return "\n".join(L)


def write_reports(report: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {"json": os.path.join(out_dir, "report.json"), "csv": os.path.join(out_dir, "measurements.csv"),
             "markdown": os.path.join(out_dir, "summary.md")}
    atomic_write_json(paths["json"], report)
    with open(paths["csv"], "w", newline="") as fh:
        fh.write(render_csv(report))
    with open(paths["markdown"], "w") as fh:
        fh.write(render_markdown(report))
    return paths
