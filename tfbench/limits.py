"""Resource limits and a low-overhead watchdog.

Design
------
* ``check_limits`` is a pure function (easy to unit test).
* ``ResourceMonitor`` samples the OS from a *background thread* (default every
  0.5 s).  Timed code never calls psutil; it only reads an ``Event`` flag
  (``monitor.aborted``), which costs nanoseconds.
* The monitor cannot interrupt a native call that is blocked inside Metal /
  TensorFlow.  Two escalation layers exist for that case:
    1. ``on_hard_stop`` (worker side): after ``hard_grace_s`` with the abort
       unacknowledged, partial results are flushed and the process exits.
    2. The supervisor process (cli.py) independently SIGTERMs / SIGKILLs a
       worker that stays over a limit or over its runtime budget.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import psutil

GB = 1024 ** 3
MB = 1024 ** 2


@dataclass(frozen=True)
class ResourceLimits:
    max_extra_gb: float = 6.0        # extra system memory vs. the job's baseline
    min_avail_gb: float = 2.0        # absolute floor of available memory
    max_sys_pct: float = 80.0        # absolute ceiling of system memory use
    max_swap_mb: float = 512.0       # swap growth vs. baseline
    max_runtime_s: float = 20 * 60.0  # per benchmark (test)

    def problems(self) -> List[str]:
        out = []
        if self.max_extra_gb <= 0:
            out.append("--max-gb must be > 0")
        if self.min_avail_gb < 0:
            out.append("--min-avail-gb must be >= 0")
        if not 0 < self.max_sys_pct <= 100:
            out.append("--max-sys-pct must be in (0, 100]")
        if self.max_swap_mb < 0:
            out.append("--max-swap-mb must be >= 0")
        if self.max_runtime_s <= 0:
            out.append("--max-runtime-min must be > 0")
        return out

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MemSample:
    t: float
    avail_gb: float
    sys_pct: float
    swap_used_gb: float
    rss_gb: float = 0.0


@dataclass(frozen=True)
class Baseline:
    t0: float
    avail_gb: float
    swap_used_gb: float


class LimitExceeded(RuntimeError):
    """Raised in the worker when a resource limit (or termination request) fired."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_PROC = psutil.Process()


def take_sample() -> MemSample:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return MemSample(
        t=time.monotonic(),
        avail_gb=vm.available / GB,
        sys_pct=vm.percent,
        swap_used_gb=sw.used / GB,
        rss_gb=_PROC.memory_info().rss / GB,
    )


def capture_baseline(sample: Optional[MemSample] = None) -> Baseline:
    s = sample or take_sample()
    return Baseline(t0=s.t, avail_gb=s.avail_gb, swap_used_gb=s.swap_used_gb)


def extra_gb(base: Baseline, s: MemSample) -> float:
    return base.avail_gb - s.avail_gb


def swap_growth_mb(base: Baseline, s: MemSample) -> float:
    return (s.swap_used_gb - base.swap_used_gb) * 1024.0


def check_limits(limits: ResourceLimits, base: Baseline, s: MemSample,
                 runtime_budget_s: Optional[float] = None) -> Optional[str]:
    """Return a human-readable violation, or None.  Boundaries are inclusive."""
    budget = limits.max_runtime_s if runtime_budget_s is None else runtime_budget_s
    elapsed = s.t - base.t0
    if elapsed >= budget:
        return f"runtime budget exhausted ({elapsed:.0f}s >= {budget:.0f}s)"
    extra = extra_gb(base, s)
    if extra >= limits.max_extra_gb:
        return f"extra system memory {extra:.2f} GB >= limit {limits.max_extra_gb:.2f} GB"
    if s.avail_gb < limits.min_avail_gb:
        return f"available memory {s.avail_gb:.2f} GB < floor {limits.min_avail_gb:.2f} GB"
    if s.sys_pct >= limits.max_sys_pct:
        return f"system memory use {s.sys_pct:.1f}% >= limit {limits.max_sys_pct:.1f}%"
    swap = swap_growth_mb(base, s)
    if swap >= limits.max_swap_mb:
        return f"swap grew {swap:.0f} MB >= limit {limits.max_swap_mb:.0f} MB"
    return None


def memory_headroom_gb(limits: ResourceLimits, s: MemSample, total_gb: Optional[float] = None) -> float:
    """Additional memory a benchmark may still use right now: the SMALLEST of the three memory limits given current usage
    (extra-memory allowance, available-memory floor, system-use ceiling).  Planning against this, instead of only
    --max-gb, keeps estimated peaks inside the limit that would actually fire first."""
    total = total_gb if total_gb is not None else psutil.virtual_memory().total / GB
    by_extra = limits.max_extra_gb
    by_floor = s.avail_gb - limits.min_avail_gb
    by_pct = total * limits.max_sys_pct / 100.0 - total * s.sys_pct / 100.0
    return max(0.0, min(by_extra, by_floor, by_pct))


def preflight(limits: ResourceLimits, s: MemSample) -> List[str]:
    """Absolute limits that are ALREADY violated before any work starts.

    Limits are never raised automatically: if this returns problems the caller
    must refuse to run and tell the user to free memory (or choose limits
    deliberately).
    """
    out = []
    if s.avail_gb < limits.min_avail_gb:
        out.append(f"only {s.avail_gb:.2f} GB available (floor is {limits.min_avail_gb:.2f} GB)")
    if s.sys_pct >= limits.max_sys_pct:
        out.append(f"system memory already at {s.sys_pct:.1f}% (limit {limits.max_sys_pct:.1f}%)")
    headroom = s.avail_gb - limits.min_avail_gb
    if headroom < 0.5:
        out.append(f"headroom above the availability floor is only {max(headroom, 0):.2f} GB")
    return out


class ResourceMonitor:
    """Background watchdog.  Safe to call ``poll_once`` directly in tests."""

    def __init__(self, limits: ResourceLimits, interval: float = 0.5,
                 sampler: Callable[[], MemSample] = take_sample,
                 runtime_budget_s: Optional[float] = None,
                 hard_grace_s: float = 30.0,
                 on_hard_stop: Optional[Callable[[str], None]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.interval = interval
        self._sampler = sampler
        self.runtime_budget_s = runtime_budget_s
        self.hard_grace_s = hard_grace_s
        self.on_hard_stop = on_hard_stop
        self._clock = clock
        self._abort = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.reason = ""
        self._abort_time: Optional[float] = None
        self._hard_done = False
        self.baseline = capture_baseline(self._sampler())
        self.last: MemSample = self._sampler()
        self.n_polls = 0
        self._peaks = self._fresh_peaks()
        self.run_peaks = self._fresh_peaks()

    # ------------------------------------------------------------- peaks
    @staticmethod
    def _fresh_peaks() -> Dict[str, float]:
        return {"max_extra_gb": float("-inf"), "min_avail_gb": float("inf"),
                "max_sys_pct": float("-inf"), "max_swap_growth_mb": float("-inf"),
                "max_rss_gb": float("-inf")}

    def _update(self, peaks: Dict[str, float], s: MemSample) -> None:
        peaks["max_extra_gb"] = max(peaks["max_extra_gb"], extra_gb(self.baseline, s))
        peaks["min_avail_gb"] = min(peaks["min_avail_gb"], s.avail_gb)
        peaks["max_sys_pct"] = max(peaks["max_sys_pct"], s.sys_pct)
        peaks["max_swap_growth_mb"] = max(peaks["max_swap_growth_mb"], swap_growth_mb(self.baseline, s))
        peaks["max_rss_gb"] = max(peaks["max_rss_gb"], s.rss_gb)

    def window_peaks(self, reset: bool = True) -> Dict[str, float]:
        """Peaks since the previous call (per generation / per repeat)."""
        with self._lock:
            out = dict(self._peaks)
            if reset:
                self._peaks = self._fresh_peaks()
        return out

    # ------------------------------------------------------------- control
    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    def request_abort(self, reason: str) -> None:
        with self._lock:
            if not self._abort.is_set():
                self.reason = reason
                self._abort_time = self._clock()
                self._abort.set()

    def raise_if_aborted(self) -> None:
        if self._abort.is_set():
            raise LimitExceeded(self.reason)

    def poll_once(self) -> Optional[str]:
        s = self._sampler()
        self.n_polls += 1
        with self._lock:
            self.last = s
            self._update(self._peaks, s)
            self._update(self.run_peaks, s)
        if not self._abort.is_set():
            reason = check_limits(self.limits, self.baseline, s, self.runtime_budget_s)
            if reason:
                self.request_abort(reason)
        elif (self.on_hard_stop and not self._hard_done and self._abort_time is not None
              and self._clock() - self._abort_time >= self.hard_grace_s):
            self._hard_done = True
            self.on_hard_stop(self.reason)
        return self.reason or None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.poll_once()
            except Exception:  # a failing sampler must never kill the benchmark
                pass

    def start(self) -> "ResourceMonitor":
        if self._thread is None:
            try:
                self.poll_once()
            except Exception:       # a failing sampler must never prevent the benchmark from starting
                pass
            self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
