"""Experimental CPU/GPU execution-policy ADVISOR (recommendation mode only).

It NEVER changes the device of a running benchmark.  It predicts population evaluation time per strategy from a cost model
calibrated on MEASURED components:

    seconds per evaluation  =  fixed(device, layers)  +  steps * step(device, params * batch)

(fixed = construct + trace + first execution + fitness excluding the first step; step = training seconds per step).
Strategy gains for multiprocessing / batching are applied as MEASURED multipliers.  To avoid overfitting a single machine or
session the model is validated by leave-one-tier-out prediction error, it flags extrapolation outside the calibrated range, and
it reports the machine it was calibrated on.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .workload import Point, Settings, n_evaluations, point_param_stats

MIN_S = 1e-6


def _cell_features(r: Dict[str, Any], st: Settings) -> Optional[Dict[str, float]]:
    if r.get("status") != "completed" or r["strategy"]["mode"] != "seq" or r["point"]["mode"] != "hetero":
        return None
    if r["point"]["steps"] < 2 or not r.get("evaluations") or not r["comps"].get("train"):
        return None
    if r["strategy"]["device"] == "gpu" and r.get("placement") == "soft":
        return None
    c, e, s = r["comps"], r["evaluations"], r["point"]["steps"]
    step = c["train"] / (e * (s - 1))
    fixed = max(0.0, (c["construct"] + c["trace"] + c["first_exec"] + c["fitness"] + c.get("teardown", 0.0)) / e - step)
    ps = point_param_stats(Point(**r["point"]), st)
    return {"device": r["strategy"]["device"], "tier": r["point"]["tier"], "layers": ps["layers_mean"], "params": ps["mean"],
            "steps": s, "fixed": fixed, "step": step, "work": ps["mean"] * st.batch}


def _lin(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    if len(set(xs)) < 2:
        return float(np.mean(ys)), 0.0
    a = np.vstack([np.ones(len(xs)), np.array(xs)]).T
    coef, *_ = np.linalg.lstsq(a, np.array(ys), rcond=None)
    return float(coef[0]), float(coef[1])


def fit(results: List[Dict[str, Any]], st: Settings, exclude_tier: Optional[str] = None) -> Dict[str, Any]:
    feats = [f for r in results if (f := _cell_features(r, st)) and f["tier"] != exclude_tier]
    model: Dict[str, Any] = {"devices": {}}
    for dev in ("cpu", "gpu"):
        fs = [f for f in feats if f["device"] == dev]
        if len(fs) < 2:
            continue
        f0, f1 = _lin([f["layers"] for f in fs], [f["fixed"] for f in fs])
        s0, s1 = _lin([f["work"] for f in fs], [f["step"] for f in fs])
        model["devices"][dev] = {"fixed": [f0, f1], "step": [s0, s1], "n_cells": len(fs),
                                 "tiers": sorted({f["tier"] for f in fs}),
                                 "layers_range": [min(f["layers"] for f in fs), max(f["layers"] for f in fs)],
                                 "params_range": [min(f["params"] for f in fs), max(f["params"] for f in fs)],
                                 "steps_range": [min(f["steps"] for f in fs), max(f["steps"] for f in fs)]}
    return model


def predict_eval_s(model: Dict[str, Any], device: str, layers: float, params: float, batch: int, steps: int) -> float:
    d = model["devices"][device]
    fixed = max(MIN_S, d["fixed"][0] + d["fixed"][1] * layers)
    step = max(MIN_S, d["step"][0] + d["step"][1] * params * batch)
    return fixed + steps * step


def strategy_gains(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Measured median speed-ups of mp / batched over sequential at identical Points (exploration + confirmation)."""
    seq: Dict[Tuple[str, str], List[float]] = {}
    for r in results:
        if r.get("status") == "completed" and r["strategy"]["mode"] == "seq" and (r.get("ips") or 0) > 0:
            seq.setdefault((Point(**r["point"]).key, r["strategy"]["device"]), []).append(r["ips"])
    gains: Dict[str, List[float]] = {}
    for r in results:
        s = r["strategy"]
        if r.get("status") != "completed" or s["mode"] == "seq" or not (r.get("ips") or 0) > 0:
            continue
        if s["device"] == "gpu" and r.get("placement") == "soft":
            continue
        base = seq.get((Point(**r["point"]).key, s["device"]))
        if base:
            name = f'{s["device"]}-mp{s["workers"]}' if s["mode"] == "mp" else f'{s["device"]}-batched-{r["point"]["mode"]}'
            gains.setdefault(name, []).append(r["ips"] / statistics.median(base))
    return {k: {"median": statistics.median(v), "min": min(v), "max": max(v), "n": len(v)} for k, v in gains.items()}


def build_model(results: List[Dict[str, Any]], st: Settings, machine: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    m = fit(results, st)
    m["gains"] = strategy_gains(results)
    m["batch"] = st.batch
    m["calibrated_on"] = machine or {}
    m["kind"] = "estimated"
    return m


def validate(results: List[Dict[str, Any]], st: Settings) -> Dict[str, Any]:
    """Leave-one-tier-out: refit without a tier and predict the wall time of that tier's sequential cells."""
    out: Dict[str, Any] = {"method": "leave-one-tier-out on measured sequential cells (S >= 2)", "per_device": {}}
    tiers = sorted({r["point"]["tier"] for r in results})
    for dev in ("cpu", "gpu"):
        errs: List[float] = []
        per_tier: Dict[str, float] = {}
        for t in tiers:
            m = fit(results, st, exclude_tier=t)
            if dev not in m["devices"]:
                continue
            e_t = []
            for r in results:
                f = _cell_features(r, st)
                if not f or f["device"] != dev or f["tier"] != t:
                    continue
                n_ev = r["evaluations"]
                pred = n_ev * predict_eval_s(m, dev, f["layers"], f["params"], st.batch, f["steps"])
                actual = r["wall_s"]
                if actual > 0:
                    e_t.append(abs(pred - actual) / actual)
            if e_t:
                per_tier[t] = float(np.mean(e_t))
                errs.extend(e_t)
        if errs:
            out["per_device"][dev] = {"mape": float(np.mean(errs)), "per_held_out_tier": per_tier, "n_cells": len(errs)}
    mapes = [v["mape"] for v in out["per_device"].values()]
    out["reliable"] = bool(mapes) and max(mapes) <= 0.30 and len(out["per_device"]) == 2
    out["note"] = ("Few tiers means a coarse test; 'reliable' = held-out MAPE <= 30% on both devices. An unreliable advisor still "
                   "reports its predictions but always with confidence 'low'.")
    return out


def recommend(model: Dict[str, Any], *, params: float, layers: float, steps: int, batch: int, population: int,
              generations: int = 3, available_gb: Optional[float] = None, homogeneous: bool = False,
              validation: Optional[Dict[str, Any]] = None, machine: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Recommendation only.  Returns predicted seconds per strategy, a pick, a confidence and every caveat."""
    warnings: List[str] = []
    pred: Dict[str, float] = {}
    extrap = False
    n_ev = n_evaluations(population, generations)
    for dev in ("cpu", "gpu"):
        d = model["devices"].get(dev)
        if not d:
            warnings.append(f"no calibration for {dev}: it is excluded from the recommendation")
            continue
        for name, v, (lo, hi) in (("layers", layers, d["layers_range"]), ("params", params, d["params_range"]), ("steps", steps, d["steps_range"])):
            if v < lo / 2 or v > hi * 2:
                extrap = True
                warnings.append(f"{dev}: {name}={v:g} lies more than 2x outside the calibrated range [{lo:g}, {hi:g}] (extrapolation)")
        pred[f"{dev}-seq"] = n_ev * predict_eval_s(model, dev, layers, params, batch, steps)
    gains = model.get("gains", {})
    for name, g in gains.items():
        dev = name.split("-")[0]
        base = pred.get(f"{dev}-seq")
        if base is None:
            continue
        if "-mp" in name:
            pred[name] = base / g["median"] + 8.0                      # + worker-pool start-up (spawn + TensorFlow import)
            warnings.append(f"{name} includes ~8 s of one-time pool start-up")
        elif name.endswith("homo") == homogeneous:
            pred[name.replace("-hetero", "").replace("-homo", "")] = base / g["median"]
            warnings.append(f"{name} gain is a single measured multiplier ({g['median']:.2f}x, range {g['min']:.2f}-{g['max']:.2f}, n={g['n']}); it is not modelled")
    if not pred:
        return {"mode": "recommendation-only", "recommendation": None, "reason": "model has no calibrated device", "warnings": warnings}
    mem = params * 16 / 1024 ** 3 * (population if homogeneous else 1) + 1.0
    mem_ok = available_gb is None or mem <= 0.5 * available_gb
    if not mem_ok:
        warnings.append(f"estimated memory {mem:.1f} GB exceeds half of the available {available_gb:.1f} GB: prefer sequential strategies")
        for k in [k for k in pred if "batched" in k]:
            pred[k] = float("inf")
    ranked = sorted(pred.items(), key=lambda kv: kv[1])
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    margin = (second[1] / best[1]) if second and best[1] > 0 else float("inf")
    cal = model.get("calibrated_on") or {}
    if machine and cal and machine.get("chip") and cal.get("chip") and machine["chip"] != cal["chip"]:
        warnings.append(f"calibrated on '{cal.get('chip')}' but queried for '{machine.get('chip')}': re-run the scaling benchmark on this machine")
        extrap = True
    reliable = bool(validation and validation.get("reliable"))
    if extrap or not reliable or margin < 1.2:
        conf = "low"
    elif margin < 1.5:
        conf = "medium"
    else:
        conf = "high"
    if not reliable:
        warnings.append("advisor failed or lacks leave-one-tier-out validation: confidence capped at 'low'")
    if margin < 1.2:
        warnings.append(f"top two strategies are within {100 * (margin - 1):.0f}% of each other: too close to call")
    return {"mode": "recommendation-only", "recommendation": best[0], "confidence": conf, "predicted_seconds": dict(ranked),
            "margin_over_second": margin, "extrapolating": extrap, "memory_ok": mem_ok, "warnings": warnings,
            "note": "Advice only: the benchmark never switches devices on its own."}
