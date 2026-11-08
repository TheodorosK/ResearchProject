#!/usr/bin/env python3
"""Compact plain-text digest of the `scaling` job in a report.json (what a human wants to read first).

    python tools/scaling_digest.py bench_results/RUN/report.json
"""
import json
import sys


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def main(path: str) -> int:
    with open(path) as fh:
        rep = json.load(fh)
    job = next((j for j in rep["jobs"] if j.get("test") == "scaling"), None)
    if not job or not job.get("data", {}).get("analysis"):
        print("no scaling analysis in this report")
        return 2
    d, an = job["data"], job["data"]["analysis"]
    ctl = d["controller"]
    cells = ctl["cells"]
    done = [c for c in cells if c["status"] == "completed"]
    hw, sw = rep.get("hardware", {}), rep.get("software", {})
    print(f"== {hw.get('chip')} | {hw.get('ram_total_gb', 0):.0f} GB | TF {sw.get('tensorflow')} | metal {sw.get('tensorflow-metal')} | grappler {rep.get('config', {}).get('grappler')} ==")
    print(f"status: {job['status']} {job.get('reason', '')} | elapsed {ctl['elapsed_s'] / 60:.1f} min | cells completed {len(done)}, not completed {len(cells) - len(done)}, "
          f"skipped {len(ctl['skipped'])} | blocks {len(ctl['blocks'])}")
    bad_blocks = [b for b in ctl["blocks"] if b["status"] != "completed"]
    for b in bad_blocks:
        print(f"  BLOCK {b['block_id']}: {b['status']} - {b['reason'][:120]}")
    for c in cells:
        if c["status"] != "completed":
            print(f"  CELL {c['cell_id']}: {c['status']} - {c.get('reason', '')[:100]}")
    for s in ctl["skipped"]:
        print(f"  SKIPPED {s['cell_id']}: {s['reason'][:100]}")

    print("\n-- sequential CPU vs GPU (identical strategy; ratio = GPU/CPU individuals per second) --")
    print(f"{'point':<26}{'explore':>9}{'confirmed [95% CI]':>26}{'reps G/C':>10}{'CPU ind/s':>11}{'GPU ind/s':>11}  verdict")
    for p in an["seq_points"]:
        c = p["confirm"]
        conf = f"{c['ratio']:.3g} [{c['lo']:.3g}, {c['hi']:.3g}]" if c.get("lo") is not None else "n/a"
        print(f"{p['key']:<26}{f(p['explore_ratio']):>9}{conf:>26}{str(c['n_num']) + '/' + str(c['n_den']):>10}"
              f"{f((c.get('cpu') or {}).get('median') or p['explore_cpu_ips']):>11}{f((c.get('gpu') or {}).get('median') or p['explore_gpu_ips']):>11}  {p['verdict']}")
    cr = an["crossover"]
    print(f"crossover status: {cr['status']} | confirmed points {cr['n_confirmed_points']}/{cr['n_points']} | best ratio "
          f"{f(cr['best']['ratio']) if cr.get('best') else 'n/a'} at {cr['best']['key'] if cr.get('best') else 'n/a'}")
    for e in cr["confirmed_crossings"]:
        print(f"  CONFIRMED crossing along {e['axis']}: {e['between']}")
    for e in cr["candidate_crossings"]:
        print(f"  candidate (unconfirmed) along {e['axis']}: {e['between']}")

    bp = [p for p in an.get("batched_points", []) if p["ratio"] is not None]
    if bp:
        print("\n-- batched CPU vs GPU (identical strategy; ratio = GPU/CPU) --")
        for p in bp:
            c = p["confirm"]
            conf = f"{c['ratio']:.3g} [{c['lo']:.3g}, {c['hi']:.3g}]" if c.get("lo") is not None else "n/a"
            print(f"{p['key']:<28}explore {f(p['explore_ratio']):>6}  confirmed {conf:<22} {p['verdict']}")
        cb = cr.get("batched", {})
        for e in cb.get("confirmed_crossings", []):
            print(f"  CONFIRMED batched crossing along population: P {e['between']} ({e['tier']}, S={e['steps']}, {e['population_mode']})"
                  + (f", interpolated ~P={e['interpolated_x']:.2g} (rough)" if e.get("interpolated_x") else ""))
        for e in cb.get("candidate_crossings", []):
            print(f"  candidate (unconfirmed) batched crossing along population: P {e['between']} ({e['tier']}, S={e['steps']}, {e['population_mode']})")
    print("\n-- strategies vs sequential on the same device and workload (cross-strategy) --")
    for r in an["strategies"]:
        ci = f"[{r['ci_lo']:.3g}, {r['ci_hi']:.3g}]" if r.get("ci_lo") else "n/a"
        print(f"{r['point']:<26}{r['strategy']:<14}{f(r['speedup_vs_sequential']):>7}x {ci:<16}{r['evidence']:<22}"
              f"batchable={f(r['batched_member_fraction'])} padding={f(r['padded_ratio'])} mp_eff={f(r['parallel_efficiency'])}")
    if an["best_available"]:
        print("best strategy per device (cross-strategy, exploration):")
        for b in an["best_available"]:
            print(f"  {b['point']} [{b.get('population_mode', '')}]: CPU {b['best_cpu']} {f(b['best_cpu_ips'])} ind/s vs GPU {b['best_gpu']} {f(b['best_gpu_ips'])} ind/s -> {f(b['ratio_best_gpu_over_best_cpu'])}x" + ("" if b.get("cpu_multiprocessing_measured") else "  [CPU multiprocessing NOT measured]"))

    print("\n-- where the time goes (sequential, share of wall time) --")
    for c in an["components"]:
        s = c["share"]
        print(f"{c['device']}/{c['tier']:<7} construct {s['construct']:.0%} trace {s['trace']:.0%} first_exec {s['first_exec']:.0%} train {s['train']:.0%} "
              f"fitness {s['fitness']:.0%} teardown {s.get('teardown', 0):.0%} ga {s['ga']:.1%} cleanup {s['cleanup']:.1%} other {s['unattributed']:.1%} (wall {c['wall_s']:.0f}s, {c['evaluations']} evals)")
    print("\n-- cost model (ESTIMATED) --")
    for t, b in an["cost_model"]["break_even"].items():
        be = f"{b['break_even_steps']:.0f} steps" if b["break_even_steps"] else (b["reason_none"] or "n/a")
        print(f"{t}: GPU {b['gpu_fixed_s']:.2f}s/model + {b['gpu_step_s'] * 1000:.2f} ms/step | CPU {b['cpu_fixed_s']:.2f}s/model + {b['cpu_step_s'] * 1000:.2f} ms/step | break-even: {be}")
    mem = an["memory"]
    if mem:
        top = sorted(mem, key=lambda m: -m["peak_additional_gb"])[:3]
        print("\n-- memory (largest additional) --")
        for m in top:
            print(f"{m['strategy']:<12}{m['tier']:<8}{m['population_mode']:<8}P={m['population']:<3}{m['peak_additional_gb']:.2f} GB  retained {f(m['retained_gb'])} GB  swap {m['swap_growth_mb']:.0f} MB")
    sv = an["scheduler_validation"]
    print("\n-- advisor --  held-out MAPE: " + ", ".join(f"{k} {v['mape']:.0%}" for k, v in sv["per_device"].items()) + f" | reliable: {sv['reliable']}")
    print("\n-- answers --")
    for a in an["answers"][:2]:
        print(f"* {a['q']} [{a['confidence']}]\n  {a['a'][:420]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
