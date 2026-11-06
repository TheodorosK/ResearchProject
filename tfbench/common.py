"""Shared helpers: measurement records, timing statistics and JSON sanitising.

Every number the suite reports carries a ``kind``:
  measured   - read directly from a clock / OS counter / TensorFlow
  calculated - derived arithmetically from measured values (medians, ratios)
  estimated  - an analytical model or heuristic, NOT observed
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from typing import Any, Dict, Iterable, List, Optional

KINDS = ("measured", "calculated", "estimated")


def timing_stats(values: Iterable[float]) -> Dict[str, float]:
    """Median / mean / sample stdev / coefficient of variation / min / max."""
    vals = [float(v) for v in values]
    if not vals:
        return {}
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": mean,
        "stdev": sd,
        "cv": (sd / mean) if mean else float("nan"),
        "min": min(vals),
        "max": max(vals),
    }


def sanitize(obj: Any) -> Any:
    """Make an object strictly JSON-safe (NaN/inf -> None, numpy -> python)."""
    try:
        import numpy as np  # local import: keeps this module numpy-optional
        if isinstance(obj, np.generic):
            obj = obj.item()
        elif isinstance(obj, np.ndarray):
            obj = obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import threading
    tmp = f"{path}.tmp{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as fh:
        json.dump(sanitize(payload), fh, indent=2, allow_nan=False)
    os.replace(tmp, path)


class Recorder:
    """Collects measurements, structured data and notes for one job."""

    def __init__(self, job: str, path: Optional[str] = None):
        self.job = job
        self.path = path
        self.status = "running"        # running | completed | aborted | failed | unsupported | placement_failed | interrupted
        self.reason = ""
        self.measurements: List[Dict[str, Any]] = []
        self.data: Dict[str, Any] = {}
        self.notes: List[str] = []
        self.started = time.time()

    def measure(self, name: str, value: Any, unit: str = "", kind: str = "measured",
                **context: Any) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown measurement kind {kind!r}")
        self.measurements.append(
            {"name": name, "value": value, "unit": unit, "kind": kind, "context": context})

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def set_status(self, status: str, reason: str = "") -> None:
        self.status, self.reason = status, reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job": self.job,
            "status": self.status,
            "reason": self.reason,
            "wall_s": time.time() - self.started,
            "measurements": self.measurements,
            "data": self.data,
            "notes": self.notes,
        }

    def write_final(self, monitor=None) -> None:
        """Attach resource-monitor facts (if any) and persist."""
        if monitor is not None:
            self.data["resource_monitor"] = {"run_peaks": monitor.run_peaks, "polls": monitor.n_polls,
                                             "limit_reason": monitor.reason or None, "interval_s": monitor.interval}
        self.checkpoint()

    def checkpoint(self) -> None:
        """Persist partial results (called after each repeat/generation)."""
        if self.path:
            atomic_write_json(self.path, self.to_dict())


# ---------------------------------------------------------------- memory-growth analysis
def linear_slope(y) -> float:
    """Least-squares slope of y against 0..n-1 (units of y per index)."""
    y = [float(v) for v in y]
    n = len(y)
    if n < 2:
        return 0.0
    xm, ym = (n - 1) / 2.0, sum(y) / n
    den = sum((i - xm) ** 2 for i in range(n))
    return sum((i - xm) * (v - ym) for i, v in enumerate(y)) / den


def classify_memory_growth(series_mb, python_mb=None, warmup: int = 3, slope_threshold_mb: float = 1.0):
    """Heuristic classification of per-cycle memory growth.  NOT proof of a leak.

    bounded                 post-warm-up slope below threshold (warm-up/allocator caching, no growth)
    saturating              growth present but decelerating (2nd-half slope < 50 % of 1st half)
    python_retention        growth tracked by Python allocations (tracemalloc): references are retained
    continued_native_growth growth continues at >= 50 % of its early rate with flat Python memory:
                            potential native leak OR an unbounded cache -- needs a longer run
    inconclusive            fewer than 6 post-warm-up cycles
    """
    s = [float(v) for v in series_mb][warmup:]
    out = {"warmup_cycles": warmup, "threshold_mb_per_cycle": slope_threshold_mb, "n_used": len(s)}
    if len(s) < 6:
        out.update(verdict="inconclusive", reason="need >= 6 post-warm-up cycles")
        return out
    h = len(s) // 2
    slope, s1, s2 = linear_slope(s), linear_slope(s[:h]), linear_slope(s[h:])
    py_slope = linear_slope([float(v) for v in python_mb][warmup:]) if python_mb else None
    out.update(slope_mb_per_cycle=slope, first_half_slope=s1, second_half_slope=s2,
               python_slope_mb_per_cycle=py_slope, total_growth_mb=s[-1] - s[0])
    if slope < slope_threshold_mb:
        out.update(verdict="bounded", reason="no sustained growth after warm-up (consistent with allocator caching)")
    elif py_slope is not None and py_slope >= 0.5 * slope:
        out.update(verdict="python_retention", reason="growth is mirrored by Python-level allocations: objects are still referenced")
    elif s2 >= 0.5 * s1 and s2 >= slope_threshold_mb:
        out.update(verdict="continued_native_growth",
                   reason="growth continues at a similar rate with flat Python memory: potential native leak or "
                          "unbounded cache; repeat with more cycles before concluding")
    else:
        out.update(verdict="saturating", reason="growth is decelerating (consistent with a cache filling up)")
    return out
