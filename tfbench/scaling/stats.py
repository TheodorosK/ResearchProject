"""Small-sample statistics for paired CPU/GPU comparisons.  Pure Python (no scipy).

Throughput ratios are analysed in log space: log r = mean(log gpu) - mean(log cpu), with a Welch (unequal variance)
t-interval.  A crossover is only declared from CONFIRMED verdicts (confidence interval excludes parity)."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

# two-sided 95 % Student-t critical values
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042, 60: 2.000}
_T90 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
        12: 1.782, 15: 1.753, 20: 1.725, 30: 1.697, 60: 1.671}


def t_crit(df: float, conf: float = 0.95) -> float:
    table = _T975 if conf >= 0.95 else _T90
    inf = 1.960 if conf >= 0.95 else 1.645
    if df < 1:
        return float("inf")
    keys = sorted(table)
    if df >= keys[-1]:
        return inf if df > 200 else table[keys[-1]]
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return table[lo]
    w = (df - lo) / (hi - lo)
    return table[lo] * (1 - w) + table[hi] * w


def median_cv(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(x) for x in xs]
    if not xs:
        return {"n": 0, "median": None, "cv": None}
    m = statistics.fmean(xs)
    cv = (statistics.stdev(xs) / m) if len(xs) > 1 and m else None
    return {"n": len(xs), "median": statistics.median(xs), "mean": m, "cv": cv, "min": min(xs), "max": max(xs)}


def log_ratio_ci(num: Sequence[float], den: Sequence[float], conf: float = 0.95) -> Dict[str, Any]:
    """Geometric-mean ratio num/den with a Welch confidence interval.  Needs >= 2 samples on EACH side for a CI."""
    num = [x for x in num if x and x > 0]
    den = [x for x in den if x and x > 0]
    out: Dict[str, Any] = {"n_num": len(num), "n_den": len(den), "ratio": None, "lo": None, "hi": None, "df": None}
    if not num or not den:
        return out
    ln, ld = [math.log(x) for x in num], [math.log(x) for x in den]
    diff = statistics.fmean(ln) - statistics.fmean(ld)
    out["ratio"] = math.exp(diff)
    if len(num) < 2 or len(den) < 2:
        return out
    vn, vd = statistics.variance(ln) / len(ln), statistics.variance(ld) / len(ld)
    se = math.sqrt(vn + vd)
    if se == 0:                                           # identical repeats: no measurable spread
        out.update(lo=out["ratio"], hi=out["ratio"], df=len(ln) + len(ld) - 2)
        return out
    df = (vn + vd) ** 2 / ((vn ** 2) / (len(ln) - 1) + (vd ** 2) / (len(ld) - 1))
    half = t_crit(df, conf) * se
    out.update(lo=math.exp(diff - half), hi=math.exp(diff + half), df=df)
    return out


def verdict(ci: Dict[str, Any]) -> str:
    """gpu_faster / cpu_faster only when the interval EXCLUDES parity; unconfirmed without a CI."""
    if ci.get("ratio") is None:
        return "no_data"
    if ci.get("lo") is None:
        return "unconfirmed"
    if ci["lo"] > 1.0:
        return "gpu_faster"
    if ci["hi"] < 1.0:
        return "cpu_faster"
    return "inconclusive"


def crossover_scan(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """points: ordered along one axis; each {x, ratio, verdict}.  Returns confirmed crossings, unconfirmed candidates and
    a log-linear interpolation of where ratio == 1 between the straddling points."""
    confirmed, candidates = [], []
    for a, b in zip(points, points[1:]):
        if a.get("ratio") is None or b.get("ratio") is None:
            continue
        straddle = (a["ratio"] - 1.0) * (b["ratio"] - 1.0) < 0
        entry = {"between": [a["x"], b["x"]], "ratios": [a["ratio"], b["ratio"]],
                 "interpolated_x": interpolate_parity(a["x"], a["ratio"], b["x"], b["ratio"]) if straddle else None}
        if a["verdict"] == "cpu_faster" and b["verdict"] == "gpu_faster":
            confirmed.append(entry)
        elif straddle:
            candidates.append(entry)
    return {"confirmed": confirmed, "candidates": candidates}


def interpolate_parity(x1: float, r1: float, x2: float, r2: float) -> Optional[float]:
    """x where log-ratio crosses 0, linear in log x (x > 0).  None when the points do not straddle parity."""
    if min(x1, x2, r1, r2) <= 0 or (math.log(r1) * math.log(r2)) > 0:
        return None
    l1, l2 = math.log(r1), math.log(r2)
    if l1 == l2:
        return None
    w = -l1 / (l2 - l1)
    return math.exp(math.log(x1) + w * (math.log(x2) - math.log(x1)))


def break_even_steps(fixed_gpu: float, step_gpu: float, fixed_cpu: float, step_cpu: float) -> Optional[float]:
    """Steps per individual where  fixed_g + S*step_g == fixed_c + S*step_c.  None unless the GPU's per-step cost is lower
    AND its fixed cost is higher (otherwise there is no crossing, or the GPU always wins)."""
    if step_cpu <= step_gpu or fixed_gpu <= fixed_cpu:
        return None
    return (fixed_gpu - fixed_cpu) / (step_cpu - step_gpu)
