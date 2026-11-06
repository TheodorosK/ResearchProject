"""Hardware / software / thermal telemetry.  Every probe is best-effort and reports
``available: False`` rather than guessing.  Absence of telemetry is NOT evidence that
no throttling or pressure occurred."""
from __future__ import annotations

import platform
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

import psutil


def run_cmd(args: List[str], timeout: float = 8.0) -> Optional[str]:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "") + (out.stderr or "") if out.returncode == 0 else None
    except Exception:
        return None


def _sysctl(key: str) -> Optional[str]:
    out = run_cmd(["sysctl", "-n", key], timeout=3)
    return out.strip() if out else None


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "macos_version": platform.mac_ver()[0] or None,
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": psutil.virtual_memory().total / 1024 ** 3,
    }
    if platform.system() == "Darwin":
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        info["model_identifier"] = _sysctl("hw.model")
        for key, name in (("hw.perflevel0.physicalcpu", "performance_cores"),
                          ("hw.perflevel1.physicalcpu", "efficiency_cores")):
            v = _sysctl(key)
            info[name] = int(v) if v and v.isdigit() else None
        sp = run_cmd(["system_profiler", "SPHardwareDataType"], timeout=20)
        if sp:
            m = re.search(r"Model Name:\s*(.+)", sp)
            info["model_name"] = m.group(1).strip() if m else None
            g = re.search(r"Total Number of Cores:\s*(.+)", sp)
            info["total_cores_text"] = g.group(1).strip() if g else None
        gpu = run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=20)
        if gpu:
            m = re.search(r"Total Number of Cores:\s*(\d+)", gpu)
            info["gpu_cores"] = int(m.group(1)) if m else None
    return info


def power_source() -> Dict[str, Any]:
    out = run_cmd(["pmset", "-g", "batt"], timeout=4)
    if not out:
        return {"available": False}
    m = re.search(r"Now drawing from '([^']+)'", out)
    return {"available": True, "source": m.group(1) if m else None, "raw": out.strip()}


def low_power_mode() -> Dict[str, Any]:
    out = run_cmd(["pmset", "-g"], timeout=4)
    if not out:
        return {"available": False}
    m = re.search(r"lowpowermode\s+(\d)", out)
    return {"available": bool(m), "enabled": (m.group(1) == "1") if m else None}


def thermal_state() -> Dict[str, Any]:
    """Parse ``pmset -g therm``.  Throttling is only *indicated* when a limit < 100 is reported."""
    out = run_cmd(["pmset", "-g", "therm"], timeout=4)
    if out is None:
        return {"available": False, "throttling_indicated": None,
                "note": "pmset not available; thermal state unknown"}
    vals = {}
    for key in ("CPU_Speed_Limit", "CPU_Scheduler_Limit", "CPU_Available_CPUs"):
        m = re.search(rf"{key}\s*=\s*(\d+)", out)
        if m:
            vals[key] = int(m.group(1))
    indicated: Optional[bool] = None
    if "CPU_Speed_Limit" in vals or "CPU_Scheduler_Limit" in vals:
        indicated = any(vals.get(k, 100) < 100 for k in ("CPU_Speed_Limit", "CPU_Scheduler_Limit"))
    return {"available": True, "values": vals, "throttling_indicated": indicated,
            "raw": out.strip()[:600],
            "note": ("pmset reported no limit values; this does NOT prove the absence of throttling "
                     "(GPU/clock telemetry needs `sudo powermetrics`)") if indicated is None else ""}


def memory_pressure_free_pct() -> Optional[int]:
    out = run_cmd(["memory_pressure"], timeout=6)
    if not out:
        return None
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def swap_counters() -> Dict[str, float]:
    sw = psutil.swap_memory()
    return {"used_gb": sw.used / 1024 ** 3, "total_gb": sw.total / 1024 ** 3,
            "sin_gb": sw.sin / 1024 ** 3, "sout_gb": sw.sout / 1024 ** 3}


def package_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata
    out: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for pkg in ("tensorflow", "tensorflow-macos", "tensorflow-cpu", "tensorflow-metal", "keras",
                "numpy", "psutil", "h5py", "protobuf"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def peak_rss_gb() -> Optional[float]:
    """Process-lifetime peak RSS (kernel high-water mark).  Excludes GPU/unified-memory buffers
    that Metal does not map into the process's resident set."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (v if platform.system() == "Darwin" else v * 1024) / 1024 ** 3
    except Exception:
        return None


class CpuMeter:
    """Process CPU-core usage and system-wide CPU % over a window (call reset() then read())."""

    def __init__(self):
        self._p = psutil.Process()
        self.reset()

    def reset(self) -> None:
        import time
        self._t = time.perf_counter()
        c = self._p.cpu_times()
        self._cpu = c.user + c.system
        psutil.cpu_percent(None)

    def read(self) -> Dict[str, float]:
        import time
        wall = max(time.perf_counter() - self._t, 1e-9)
        c = self._p.cpu_times()
        return {"process_cores_busy": (c.user + c.system - self._cpu) / wall,
                "system_cpu_pct": psutil.cpu_percent(None)}
