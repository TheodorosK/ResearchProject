"""Assemble the ``scaling`` job result from a controller run (shared by the real CLI and by tests / synthetic illustrations)."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..common import Recorder
from . import analysis, design
from .controller import Controller
from .execute import record_cell
from .workload import Settings


def run_scaling(st: Settings, max_memory_gb: float, runner: Callable, machine: Optional[Dict[str, Any]] = None,
                clock: Optional[Callable[[], float]] = None, log: Callable[[str], None] = print) -> Dict[str, Any]:
    kw = {"clock": clock} if clock else {}
    if st.design == "capacity":
        from .capacity import CapacityController
        ctrl = CapacityController(st, max_memory_gb, runner, log=log, **kw)
    else:
        ctrl = Controller(st, max_memory_gb, runner, log=log, **kw)
    interrupted: Optional[BaseException] = None
    try:
        ctrl.run()
    except BaseException as e:                              # noqa: BLE001 - keep partial results, re-raise after assembling
        interrupted = e
    return assemble(st, ctrl, machine, interrupted, max_memory_gb)


def assemble(st: Settings, ctrl: Controller, machine: Optional[Dict[str, Any]], interrupted: Optional[BaseException] = None,
             headroom_gb: float = 0.0) -> Dict[str, Any]:
    res = ctrl.result()
    an = analysis.analyze(res["cells"], st, res["skipped"])
    from . import scheduler
    an["scheduler_model"] = scheduler.build_model(res["cells"], st, machine)
    rec = Recorder("scaling")
    for c in res["cells"]:
        if c.get("point") and c.get("strategy"):
            record_cell(rec, {**c, "comps": c.get("comps") or {}})
    paired = [p for p in an["seq_points"] if p["ratio"] is not None]
    if st.design == "capacity":
        from . import capacity
        an["capacity"] = capacity.analyze_capacity(res, st, headroom_gb)
        if interrupted is None and not an["capacity"]["table"] and not an["capacity"]["resident"]:
            paired = []
        elif interrupted is None:
            paired = [1]
    if interrupted is not None:
        status, reason = "interrupted", f"{type(interrupted).__name__}: {interrupted}"
    elif paired:
        status, reason = "completed", ""
    else:
        status = "partial"
        reason = "no paired CPU/GPU sequential measurement was completed (" + (next(iter(res["unusable_strategies"].values()), "") or "see incomplete configurations") + ")"
    job = {"job": "scaling", "test": "scaling", "device": None, "workload": None, "placement": "strict", "status": status, "reason": reason,
           "measurements": rec.measurements, "notes": res["notes"], "wall_s": res["elapsed_s"],
           "data": {"settings": st.as_dict(), "controller": res, "analysis": an, "design_summary": design.design_summary(st)}}
    if interrupted is not None:
        job["_interrupted"] = interrupted
    return job
