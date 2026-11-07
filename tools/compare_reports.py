#!/usr/bin/env python3
"""Compare two or more tfbench report.json files side by side.

    python tools/compare_reports.py bench_results/RUN_A/report.json bench_results/RUN_B/report.json

The FIRST report is the reference; every other column shows value and ratio (value / reference).
It makes NO prediction about hardware you have not benchmarked: it only lines up measured numbers and warns when the
runs are not like-for-like (different software versions, graph optimizer, optimizer class, seed or workload size),
because ratios between such runs are not meaningful.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tfbench.report import val  # noqa: E402

SETTING_KEYS = ["quick", "seed", "placement", "grappler", "optimizer", "perf_mode", "perf_batch", "perf_steps",
                "perf_warmup", "perf_repeats", "perf_width", "perf_depth", "perf_pool", "ne_pop", "ne_generations",
                "ne_elite", "ne_epochs", "ne_batch", "ne_train_n", "ne_val_n", "ne_test_n", "ne_in_dim",
                "ne_max_layers", "ne_max_units", "ne_activations", "ne_workers", "ne_cleanup", "ne_inherit_weights",
                "memdiag_cycles", "memdiag_policies", "memdiag_devices"]
# Metrics that were observed to swing far more than throughput between two sessions on the same machine (one repeat, M1 Pro):
# compile time x3.7 (cold vs warm shader cache?), per-generation memory growth up to x5, peak memory up to 26 %, CV +-30 %.
# They get a wide band so noise is not reported as "better"/"worse". PROVISIONAL: widen/narrow with your own repeats.
VOLATILE = {"steady_run_s_cv", "first_execution_s", "peak_extra_system_gb_sampled", "system_memory_growth_per_generation_mb",
            "rss_growth_per_generation_mb", "memdiag_slope_system"}
VOLATILE_BAND = 1.0
SOFTWARE_KEYS = ["tensorflow", "tensorflow-metal", "keras", "python", "numpy"]
HARDWARE_KEYS = ["chip", "ram_total_gb", "macos_version", "performance_cores", "efficiency_cores", "gpu_cores"]

# (metric, higher_is_better)
METRICS = {
    "performance": [("steady_samples_per_s", True), ("steady_step_time_ms", False), ("steady_run_s_cv", False),
                    ("first_execution_s", False), ("peak_extra_system_gb_sampled", False),
                    ("swap_growth_during_steady_mb", False)],
    "neuroevolution": [("overall_individuals_per_s", True), ("total_evolution_s", False),
                       ("best_individual_test_r2", True), ("system_memory_growth_per_generation_mb", False),
                       ("rss_growth_per_generation_mb", False), ("peak_extra_system_gb_sampled", False)],
}
SIMILAR = 0.07      # default 'similar' band. In-session CV is 0.3-3 %, but one day-to-day repeat on an M1 Pro moved throughput by
                    # up to 6 % (2.3 % for the other three metrics); override with --similar once you have more repeats


def metric(job, name):
    if name.startswith("memdiag_"):
        c = (job.get("data", {}).get("classification") or {}).get("rss" if name.endswith("rss") else "system_extra", {})
        return c.get("slope_mb_per_cycle")
    return val(job, name)


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"
    return str(x)


def verdict(ratio, higher_better, similar=SIMILAR):
    if ratio is None:
        return ""
    if abs(ratio - 1) <= similar:
        return "similar"
    good = (ratio > 1) == higher_better
    return "better" if good else "worse"


def comparability(reports):
    ref = reports[0]
    warns = []
    for i, r in enumerate(reports[1:], start=1):
        for label, keys, a, b in (("software", SOFTWARE_KEYS, ref.get("software", {}), r.get("software", {})),
                                  ("setting", SETTING_KEYS, ref.get("config", {}), r.get("config", {}))):
            diffs = [f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in keys if a.get(k) != b.get(k)]
            if diffs:
                warns.append(f"report {i + 1} differs from the reference in {label}: " + "; ".join(diffs))
    return warns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", help="report.json files; the first is the reference")
    ap.add_argument("--volatile-band", type=float, default=VOLATILE_BAND,
                    help="relative band for volatile metrics (marked *): compile time, memory growth/peaks, CV")
    ap.add_argument("--similar", type=float, default=SIMILAR,
                    help=f"relative band reported as 'similar' (default {SIMILAR:.0%}; set it from your own repeat-to-repeat spread)")
    a = ap.parse_args(argv)
    if len(a.reports) < 2:
        ap.error("give at least two report.json files")
    reports = []
    for p in a.reports:
        with open(p) as fh:
            reports.append(json.load(fh))

    print("== runs ==")
    for i, (p, r) in enumerate(zip(a.reports, reports), start=1):
        hw, sw = r.get("hardware", {}), r.get("software", {})
        print(f"[{i}] {r.get('run_id')}  {hw.get('chip')}  {hw.get('ram_total_gb', 0):.0f} GB  macOS {hw.get('macos_version')}  "
              f"TF {sw.get('tensorflow')}  metal {sw.get('tensorflow-metal')}  "
              f"grappler={r.get('config', {}).get('grappler')}  incomplete={len(r.get('incomplete', []))}")
    for i, r in enumerate(reports[1:], start=1):
        diff = [f"{k}: {reports[0].get('hardware', {}).get(k)!r} -> {r.get('hardware', {}).get(k)!r}"
                for k in HARDWARE_KEYS if reports[0].get("hardware", {}).get(k) != r.get("hardware", {}).get(k)]
        if diff:
            print(f"    hardware differs [1] vs [{i + 1}]: " + "; ".join(diff))
    warns = comparability(reports)
    print("\n== comparability ==")
    print("\n".join("WARNING: " + w for w in warns) if warns else "like-for-like: same software, settings and seed.")
    if warns:
        print("Ratios below are NOT meaningful for the differing items above.")

    names = []
    for r in reports:
        for j in r["jobs"]:
            if j["job"] not in names:
                names.append(j["job"])
    cols = "".join(f"{f'[{i + 1}] value':>16}{f'[{i + 1}]/[1]':>10}" if i else f"{'[1] value':>16}" for i in range(len(reports)))
    print("\n== metrics ==")
    print(f"{'job / metric':<58}{cols}   change vs [1]")
    for name in names:
        test = name.split("@")[0].split(":")[0]
        specs = list(METRICS.get(test, []))
        if test == "memdiag":
            specs = [("memdiag_slope_rss", False), ("memdiag_slope_system", False)]
        rows = []
        for m, hib in specs:
            vals = []
            for r in reports:
                job = next((j for j in r["jobs"] if j["job"] == name and j.get("status") == "completed"), None)
                vals.append(metric(job, m) if job else None)
            if all(v is None for v in vals):
                continue
            band = a.volatile_band if m in VOLATILE else a.similar
            line = f"  {m + (' *' if m in VOLATILE else ''):<56}"
            ratios = []
            for i, v in enumerate(vals):
                ratio = (v / vals[0]) if (i and v is not None and vals[0]) else None
                ratios.append(ratio)
                line += f"{fmt(v):>16}" + (f"{fmt(ratio):>10}" if i else "")
            line += "   " + ", ".join(verdict(x, hib, band) for x in ratios[1:] if x is not None)
            rows.append(line)
        if rows:
            print(name)
            print("\n".join(rows))

    print("\n== derived CPU/GPU ratios (calculated within each run) ==")
    for i, r in enumerate(reports, start=1):
        d = r.get("derived", {})
        parts = [f"{x['workload']}: GPU {fmt(x['gpu_speedup_x'])}x" for x in d.get("cpu_vs_gpu_performance", [])]
        ne = d.get("cpu_vs_gpu_neuroevolution")
        if ne:
            parts.append(f"neuroevolution GPU/CPU throughput {fmt(ne.get('gpu_vs_cpu_throughput_x'))}x")
        print(f"[{i}] " + ("; ".join(parts) or "n/a"))
    print(f"\n* volatile metric (band +-{a.volatile_band:.0%}): compile time, memory growth/peaks and CV varied far more than "
          "throughput between two sessions of the same machine.")
    print(f"'similar' means within +-{a.similar:.0%} for the other metrics; run-to-run CV was 0.3-3 % in-session, session-to-session spread is "
          "what comparing two days of the SAME machine measures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
