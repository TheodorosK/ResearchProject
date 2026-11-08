#!/usr/bin/env python3
"""memwatch - lightweight, read-only memory / responsiveness recorder for real-world baselines (macOS first, Linux works partially).

    python tools/memwatch.py record --scenario A --note "usual apps, no ML"        # start; stop with Ctrl-C
    python tools/memwatch.py record --scenario C --track Lightroom Photoshop python
    python tools/memwatch.py summarize memwatch_results/<file>.csv
    python tools/memwatch.py compare memwatch_results/A.csv memwatch_results/B.csv memwatch_results/C.csv

Scenarios (labels only): A normal macOS use with your usual apps | B typical Python/TensorFlow work | C ML together with Lightroom Classic,
Photoshop and other normal apps.  It never starts, stops, signals or modifies any process; it only reads.  While recording, type a
line and press Enter to put a marker into the next row (e.g. "export started").

Cost: one sample every --interval s (default 5): psutil reads, `vm_stat`, `sysctl` (pressure level) and a ~5 ms `true` spawn.  The
process table is scanned every --procs-every s (default 60).  Output is buffered and flushed once a minute (~200 bytes per sample,
about 3 MB per day).  Ctrl-C / SIGTERM / --max-minutes stop it cleanly; the CSV stays valid if the process is killed, and
`summarize` rebuilds the summary from it.

Memory terms: *available* is what macOS can hand out without swapping (free + reclaimable cache); *demand* = wired + active + compressed
(approximate memory that cannot simply be dropped); *pressure level* is macOS's own indicator (1 normal, 2 warning, 4 critical).
Responsiveness indicators (not measurements of what you feel): scheduler lag = overshoot of a 50 ms sleep; spawn latency = time to run /usr/bin/true.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import psutil

GB = 1024 ** 3
BASE_COLS = ["t_iso", "elapsed_s", "avail_gb", "used_pct", "demand_gb", "wired_gb", "active_gb", "inactive_gb", "compressed_gb", "swap_used_gb",
             "swap_growth_mb", "swapin_mb_s", "swapout_mb_s", "pressure_level", "free_pct", "cpu_pct", "load1", "sched_lag_ms", "spawn_ms", "marker"]


# ------------------------------------------------------------------------------------------------ parsers (pure)
def parse_vm_stat(text: str) -> Dict[str, float]:
    """macOS ``vm_stat`` -> GB per category (empty dict when the format is not recognised)."""
    m = re.search(r"page size of (\d+) bytes", text)
    if not m:
        return {}
    ps = int(m.group(1))
    keys = {"free": "Pages free", "active": "Pages active", "inactive": "Pages inactive", "speculative": "Pages speculative",
            "wired": "Pages wired down", "compressed": "Pages occupied by compressor"}
    out = {}
    for k, label in keys.items():
        r = re.search(re.escape(label) + r":\s+(\d+)", text)
        if r:
            out[k] = int(r.group(1)) * ps / GB
    return out


def parse_memory_pressure(text: str) -> Optional[float]:
    m = re.search(r"free percentage:\s*(\d+)%", text)
    return float(m.group(1)) if m else None


def _run(cmd: Sequence[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout).stdout
    except Exception:                                                    # noqa: BLE001 - tool missing on this OS
        return ""


def percentile(vals: List[float], q: float) -> Optional[float]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ------------------------------------------------------------------------------------------------ sampling
class Sampler:
    def __init__(self, track: Sequence[str]):
        self.track = list(track)
        self._rx = [re.compile(re.escape(t), re.I) for t in self.track]
        self.t0 = time.monotonic()
        sw = psutil.swap_memory()
        self.swap0, self._sin, self._sout, self._t_prev = sw.used, sw.sin, sw.sout, self.t0
        psutil.cpu_percent(None)
        self._last_free_pct: Optional[float] = None
        self._last_pm = -1e9
        self._track_vals: Dict[str, float] = {t: 0.0 for t in self.track}
        self._last_scan = -1e9
        self.last_groups: List[Dict[str, Any]] = []

    @property
    def columns(self) -> List[str]:
        return BASE_COLS[:-1] + [f"trk_{t}_gb" for t in self.track] + ["marker"]

    def scan_processes(self, top: int = 8) -> List[Dict[str, Any]]:
        groups: Dict[str, float] = defaultdict(float)
        tr = {t: 0.0 for t in self.track}
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                rss = p.info["memory_info"].rss
                name = p.info["name"] or "?"
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
            groups[name] += rss
            for t, rx in zip(self.track, self._rx):
                if rx.search(name):
                    tr[t] += rss
        self._track_vals = {t: v / GB for t, v in tr.items()}
        self.last_groups = [{"name": n, "rss_gb": v / GB} for n, v in sorted(groups.items(), key=lambda kv: -kv[1])[:top]]
        return self.last_groups

    def sample(self, marker: str = "", scan: bool = False, pressure: bool = False) -> Dict[str, Any]:
        now = time.monotonic()
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        stat = parse_vm_stat(_run(["vm_stat"])) if sys.platform == "darwin" else {}
        dt = max(now - self._t_prev, 1e-6)
        row: Dict[str, Any] = {
            "t_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"), "elapsed_s": round(now - self.t0, 1),
            "avail_gb": vm.available / GB, "used_pct": vm.percent, "wired_gb": stat.get("wired"), "active_gb": stat.get("active"),
            "inactive_gb": stat.get("inactive"), "compressed_gb": stat.get("compressed"), "swap_used_gb": sw.used / GB,
            "swap_growth_mb": (sw.used - self.swap0) / 1024 ** 2, "swapin_mb_s": (sw.sin - self._sin) / 1024 ** 2 / dt,
            "swapout_mb_s": (sw.sout - self._sout) / 1024 ** 2 / dt, "cpu_pct": psutil.cpu_percent(None), "load1": os.getloadavg()[0], "marker": marker}
        if stat:
            row["demand_gb"] = (stat.get("wired", 0) + stat.get("active", 0) + stat.get("compressed", 0))
        else:
            row["demand_gb"] = (vm.total - vm.available) / GB
        lvl = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip() if sys.platform == "darwin" else ""
        row["pressure_level"] = int(lvl) if lvl.isdigit() else None
        if pressure and sys.platform == "darwin" and now - self._last_pm >= 30:
            self._last_free_pct, self._last_pm = parse_memory_pressure(_run(["memory_pressure"], 10)), now
        row["free_pct"] = self._last_free_pct
        t = time.perf_counter()
        time.sleep(0.05)
        row["sched_lag_ms"] = max(0.0, (time.perf_counter() - t - 0.05) * 1000)
        t = time.perf_counter()
        _run(["/usr/bin/true"], 5)
        row["spawn_ms"] = (time.perf_counter() - t) * 1000
        if scan or not self.last_groups:
            self.scan_processes()
        for tname, v in self._track_vals.items():
            row[f"trk_{tname}_gb"] = v
        self._sin, self._sout, self._t_prev = sw.sin, sw.sout, now
        return row


# ------------------------------------------------------------------------------------------------ record
def record(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    base = os.path.join(args.out_dir, f"memwatch_{args.scenario}_{stamp}")
    sampler = Sampler(args.track)
    stop = threading.Event()
    markers: "queue.Queue[str]" = queue.Queue()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    if sys.stdin and sys.stdin.isatty():
        def reader():
            for line in sys.stdin:
                if line.strip():
                    markers.put(line.strip())
        threading.Thread(target=reader, daemon=True).start()
    meta = {"scenario": args.scenario, "note": args.note, "interval_s": args.interval, "track": args.track, "platform": sys.platform,
            "total_gb": psutil.virtual_memory().total / GB, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(base + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    procs_fh = open(base + ".procs.jsonl", "a", buffering=1 << 16)
    n, last_flush, last_scan, last_warn = 0, time.monotonic(), -1e9, -1e9
    print(f"memwatch: scenario {args.scenario} -> {base}.csv | every {args.interval:g}s, Ctrl-C to stop, type a line + Enter to add a marker", flush=True)
    with open(base + ".csv", "w", newline="", buffering=1 << 16) as fh:
        w = csv.DictWriter(fh, fieldnames=sampler.columns, extrasaction="ignore")
        w.writeheader()
        deadline = time.monotonic() + args.max_minutes * 60
        while not stop.is_set() and time.monotonic() < deadline:
            t_loop = time.monotonic()
            mk = []
            while not markers.empty():
                mk.append(markers.get())
            scan = t_loop - last_scan >= args.procs_every
            row = sampler.sample("; ".join(mk), scan=scan, pressure=True)
            if scan:
                last_scan = t_loop
                procs_fh.write(json.dumps({"t": row["t_iso"], "top": sampler.last_groups}) + "\n")
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
            n += 1
            if t_loop - last_flush >= 60:
                fh.flush(); procs_fh.flush(); last_flush = t_loop
            if (row.get("pressure_level") or 1) >= 2 or row["swap_growth_mb"] > args.warn_swap_mb:
                if t_loop - last_warn >= 60:
                    print(f"  [warn] pressure level {row.get('pressure_level')}, swap grew {row['swap_growth_mb']:.0f} MB, available {row['avail_gb']:.1f} GB "
                          "(memwatch only observes; nothing is stopped)", flush=True)
                    last_warn = t_loop
            stop.wait(max(0.0, args.interval - (time.monotonic() - t_loop)))
        fh.flush()
    procs_fh.close()
    s = summarize_csv(base + ".csv", meta, base + ".procs.jsonl")
    with open(base + ".summary.json", "w") as fh:
        json.dump(s, fh, indent=2)
    print(f"\nmemwatch stopped after {n} samples.\n" + render_summary(s))
    print(f"files: {base}.csv  {base}.summary.json  {base}.procs.jsonl")
    return 0


# ------------------------------------------------------------------------------------------------ summaries
def _col(rows: List[Dict[str, str]], k: str) -> List[float]:
    out = []
    for r in rows:
        try:
            if r.get(k) not in (None, ""):
                out.append(float(r[k]))
        except ValueError:
            pass
    return out


def summarize_csv(path: str, meta: Optional[Dict[str, Any]] = None, procs_path: Optional[str] = None) -> Dict[str, Any]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    base_meta = path.replace(".csv", ".meta.json")
    if meta is None and os.path.exists(base_meta):
        with open(base_meta) as fh:
            meta = json.load(fh)
    meta = meta or {}
    if not rows:
        return {"file": path, "samples": 0, "meta": meta}
    el = _col(rows, "elapsed_s")
    first = [i for i, e in enumerate(el) if e <= 60] or [0]
    dem = _col(rows, "demand_gb")
    demand_start = statistics.median([dem[i] for i in first if i < len(dem)]) if dem else None

    def stats(k, qs=(0.5, 0.95)):
        v = _col(rows, k)
        return {"min": min(v), "max": max(v), **{f"p{int(q * 100)}": percentile(v, q) for q in qs}} if v else None
    lvl = _col(rows, "pressure_level")
    s: Dict[str, Any] = {
        "file": path, "meta": meta, "samples": len(rows), "duration_min": (el[-1] / 60) if el else None, "total_gb": meta.get("total_gb"),
        "avail_gb": stats("avail_gb", (0.05, 0.5)), "used_pct": stats("used_pct"), "demand_gb": stats("demand_gb"), "demand_start_gb": demand_start,
        "demand_increase_p95_gb": (percentile(dem, 0.95) - demand_start) if (dem and demand_start is not None) else None,
        "demand_increase_max_gb": (max(dem) - demand_start) if (dem and demand_start is not None) else None,
        "compressed_gb": stats("compressed_gb"), "swap_used_gb": stats("swap_used_gb"),
        "swap_growth_mb_final": (_col(rows, "swap_growth_mb") or [None])[-1], "swap_growth_mb_max": max(_col(rows, "swap_growth_mb") or [0]),
        "swapout_mb_s_p95": percentile(_col(rows, "swapout_mb_s"), 0.95), "swapin_mb_s_p95": percentile(_col(rows, "swapin_mb_s"), 0.95),
        "pressure_share_not_normal": (sum(1 for v in lvl if v >= 2) / len(lvl)) if lvl else None,
        "pressure_share_critical": (sum(1 for v in lvl if v >= 4) / len(lvl)) if lvl else None,
        "free_pct_min": min(_col(rows, "free_pct")) if _col(rows, "free_pct") else None,
        "cpu_pct": stats("cpu_pct"), "sched_lag_ms": stats("sched_lag_ms", (0.5, 0.95, 0.99)), "spawn_ms": stats("spawn_ms", (0.5, 0.95, 0.99)),
        "tracked_peak_gb": {k[4:-3]: max(_col(rows, k)) for k in rows[0] if k.startswith("trk_") and _col(rows, k)},
        "markers": [{"elapsed_s": r["elapsed_s"], "text": r["marker"]} for r in rows if r.get("marker")],
    }
    if procs_path is None:
        cand = path.replace(".csv", ".procs.jsonl")
        procs_path = cand if os.path.exists(cand) else None
    if procs_path and os.path.exists(procs_path):
        acc: Dict[str, List[float]] = defaultdict(list)
        with open(procs_path) as fh:
            for line in fh:
                try:
                    for g in json.loads(line)["top"]:
                        acc[g["name"]].append(g["rss_gb"])
                except (ValueError, KeyError):
                    pass
        s["top_processes_rss_gb_p95"] = sorted(({"name": k, "p95": percentile(v, 0.95)} for k, v in acc.items()), key=lambda d: -d["p95"])[:10]
    return s


def _f(v, nd=1, unit=""):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def render_summary(s: Dict[str, Any]) -> str:
    if not s.get("samples"):
        return "no samples"
    m = s.get("meta", {})
    a, d = s["avail_gb"], s["demand_gb"]
    L = [f"scenario {m.get('scenario', '?')} {('- ' + m['note']) if m.get('note') else ''} | {s['samples']} samples, {_f(s['duration_min'])} min | RAM {_f(s.get('total_gb'), 0)} GB",
         f"  available GB : min {_f(a['min'])}  p05 {_f(a['p5'])}  median {_f(a['p50'])}",
         f"  demand GB    : start {_f(s['demand_start_gb'])}  median {_f(d['p50'])}  p95 {_f(d['p95'])}  max {_f(d['max'])}  (increase over start: p95 {_f(s['demand_increase_p95_gb'])}, max {_f(s['demand_increase_max_gb'])})",
         f"  compressed GB: max {_f((s['compressed_gb'] or {}).get('max'))} | swap used max {_f((s['swap_used_gb'] or {}).get('max'), 2)} GB, growth at end {_f(s['swap_growth_mb_final'], 0)} MB (max {_f(s['swap_growth_mb_max'], 0)}), swap-out p95 {_f(s['swapout_mb_s_p95'], 2)} MB/s",
         f"  pressure     : not-normal {_f(None if s['pressure_share_not_normal'] is None else 100 * s['pressure_share_not_normal'], 1)}% of samples, critical {_f(None if s['pressure_share_critical'] is None else 100 * s['pressure_share_critical'], 1)}%, lowest free {_f(s['free_pct_min'], 0)}%",
         f"  CPU %        : mean-ish p50 {_f(s['cpu_pct']['p50'], 0)}  p95 {_f(s['cpu_pct']['p95'], 0)}",
         f"  responsiveness (indicators): scheduler lag p95 {_f(s['sched_lag_ms']['p95'], 1)} ms, p99 {_f(s['sched_lag_ms']['p99'], 1)}, max {_f(s['sched_lag_ms']['max'], 1)}; spawn p95 {_f(s['spawn_ms']['p95'], 1)} ms, max {_f(s['spawn_ms']['max'], 1)}"]
    if s.get("tracked_peak_gb"):
        L.append("  tracked process groups, peak RSS GB: " + ", ".join(f"{k} {v:.2f}" for k, v in s["tracked_peak_gb"].items()))
    if s.get("top_processes_rss_gb_p95"):
        L.append("  top processes (p95 RSS GB): " + ", ".join(f"{t['name']} {t['p95']:.2f}" for t in s["top_processes_rss_gb_p95"][:6]))
    for mk in s.get("markers", [])[:12]:
        L.append(f"  marker @{mk['elapsed_s']}s: {mk['text']}")
    return "\n".join(L)


def compare(summaries: List[Dict[str, Any]]) -> str:
    L = ["| scenario | min | demand start GB | demand p95 | demand max | increase p95 | avail min GB | swap growth MB | swap-out p95 MB/s | not-normal pressure % | CPU p95 % | sched lag p95 ms | spawn p95 ms |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        if not s.get("samples"):
            continue
        m = s["meta"]
        L.append(f"| {m.get('scenario', '?')} {m.get('note', '')[:30]} | {_f(s['duration_min'])} | {_f(s['demand_start_gb'])} | {_f(s['demand_gb']['p95'])} | {_f(s['demand_gb']['max'])} | "
                 f"{_f(s['demand_increase_p95_gb'])} | {_f(s['avail_gb']['min'])} | {_f(s['swap_growth_mb_max'], 0)} | {_f(s['swapout_mb_s_p95'], 2)} | "
                 f"{_f(None if s['pressure_share_not_normal'] is None else 100 * s['pressure_share_not_normal'], 1)} | {_f(s['cpu_pct']['p95'], 0)} | {_f(s['sched_lag_ms']['p95'], 1)} | {_f(s['spawn_ms']['p95'], 1)} |")
    by = {s["meta"].get("scenario"): s for s in summaries if s.get("samples")}
    if all(k in by for k in "ABC"):
        a, b, c = by["A"], by["B"], by["C"]
        apps = a["demand_gb"]["p95"]
        ml = b["demand_increase_p95_gb"]
        comb = c["demand_gb"]["p95"]
        L += ["", "Decomposition (demand = wired + active + compressed, p95; approximate, includes macOS itself):",
              f"* normal applications + macOS (A): {_f(apps)} GB",
              f"* ML workload increase over its own starting level (B): {_f(ml)} GB",
              f"* combined (C): {_f(comb)} GB; additive expectation A + B-increase = {_f(None if ml is None else apps + ml)} GB; "
              f"difference (interference / extra apps in C) = {_f(None if ml is None else comb - apps - ml)} GB",
              f"* C against installed RAM {_f(c.get('total_gb'), 0)} GB: peak demand {_f(c['demand_gb']['max'])} GB; swap growth max {_f(c['swap_growth_mb_max'], 0)} MB; "
              f"non-normal pressure {_f(None if c['pressure_share_not_normal'] is None else 100 * c['pressure_share_not_normal'], 1)}% of samples.",
              "A 16 GB machine is sufficient for scenario C only if swap growth and non-normal pressure stay near zero; the amount missing is what the swap/compression growth shows, not "
              "a value computed here."]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record until Ctrl-C")
    r.add_argument("--scenario", required=True, help="label, e.g. A, B or C")
    r.add_argument("--note", default="")
    r.add_argument("--interval", type=float, default=5.0)
    r.add_argument("--procs-every", type=float, default=60.0)
    r.add_argument("--track", nargs="*", default=[], help="process-name fragments whose summed RSS is recorded, e.g. Lightroom Photoshop python")
    r.add_argument("--out-dir", default="memwatch_results")
    r.add_argument("--max-minutes", type=float, default=240.0)
    r.add_argument("--warn-swap-mb", type=float, default=512.0, help="print a warning (nothing is stopped) when swap grew this much")
    s = sub.add_parser("summarize")
    s.add_argument("csv")
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("csv", nargs="+")
    a = p.parse_args(argv)
    if a.cmd == "record":
        if a.interval < 1:
            print("--interval below 1 s is refused (keeps the monitor lightweight)", file=sys.stderr)
            return 2
        return record(a)
    if a.cmd == "summarize":
        s_ = summarize_csv(a.csv)
        print(json.dumps(s_, indent=2) if a.json else render_summary(s_))
        return 0
    ss = [summarize_csv(f) for f in a.csv]
    print(compare(ss))
    return 0


if __name__ == "__main__":
    sys.exit(main())
