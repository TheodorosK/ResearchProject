"""Adaptive orchestration of the scaling experiment (supervisor side: never imports TensorFlow).

The controller decides WHAT to run (pilot -> expansion -> confirmation), groups cells into per-strategy blocks (one fresh
process each), enforces the total / per-configuration time budgets, prunes configurations that aborted, and records every
skipped / aborted / incomplete configuration.  Execution is delegated to an injectable ``runner`` so the logic is testable
without TensorFlow:

    runner(block: BlockSpec, remaining_s: float) -> {"status", "reason", "cells": [cell result dicts], "placement", "notes", "data"}
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import design as D
from .workload import Cell, Point, Settings, Strategy

DEVICE_ORDER = {"cpu": 0, "gpu": 1}
HARD_FAIL = ("unsupported", "placement_failed", "failed")


@dataclass
class BlockSpec:
    block_id: str
    device: str
    variant: str                       # seq | batched | mpN
    cells: List[Cell]
    phase: str
    rep: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)      # variant-specific parameters (resident probe)


class Controller:
    def __init__(self, st: Settings, max_memory_gb: float, runner: Callable[[BlockSpec, float], Dict[str, Any]],
                 clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = print,
                 cost: Optional[D.CostModel] = None):
        self.st, self.max_memory_gb, self.runner, self.clock, self.log = st, max_memory_gb, runner, clock, log
        self.cost = cost or D.CostModel()
        self.results: List[Dict[str, Any]] = []
        self.blocks: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.unusable: Dict[str, str] = {}               # strategy name -> reason (device missing / numerics failed)
        self.blocked: Dict[Tuple[str, str], float] = {}  # (strategy, tier) -> predicted cost of the cheapest aborted cell
        self.phase_log: Dict[str, Dict[str, Any]] = {}
        self.status, self.reason = "running", ""
        self._n_blocks = 0
        self.t0 = self.clock()

    # ------------------------------------------------------------------------------------ helpers
    def elapsed(self) -> float:
        return self.clock() - self.t0

    def remaining(self) -> float:
        return self.st.max_total_s - self.elapsed()

    def _skip(self, cell: Cell, reason: str) -> None:
        self.skipped.append({"cell_id": cell.cell_id, "point": cell.point.as_dict(), "strategy": cell.strategy.name,
                             "phase": cell.phase, "rep": cell.rep, "status": "skipped", "reason": reason})

    def _unusable_reason(self, strategy: Strategy) -> Optional[str]:
        return self.unusable.get(strategy.name) or self.unusable.get(f"{strategy.device}-*")

    def _mark_unusable(self, dev: str, name: str, status: str, reason: str) -> str:
        """A missing device or failed numerical/placement VALIDATION disables every strategy on that device (they share the
        same execution path); a strategy-specific failure (e.g. one unsupported kernel) stays local to that strategy."""
        device_wide = status == "unsupported" or (status == "failed" and (
            reason.startswith("numerical validation") or reason.startswith("device placement could not be verified")))
        key = f"{dev}-*" if device_wide else name
        self.unusable[key] = f"{status}: {reason[:140]}"
        return key

    def _predicted(self, cell: Cell) -> float:
        return self.cost.predict(cell.point, cell.strategy, self.st)

    def _group(self, cells: List[Cell], rep: int) -> List[Tuple[str, str, List[Cell]]]:
        groups: Dict[Tuple[str, str], List[Cell]] = {}
        for c in cells:
            groups.setdefault((c.strategy.device, c.strategy.variant), []).append(c)
        order = sorted(groups, key=lambda k: ((DEVICE_ORDER[k[0]] + rep) % 2, k[1]))      # alternate device order per repetition
        return [(d, v, sorted(groups[(d, v)], key=self._predicted)) for d, v in order]

    # ------------------------------------------------------------------------------------ core
    def run_cells(self, cells: List[Cell], phase: str, rep: int, cap_s: float) -> None:
        runnable: List[Cell] = []
        for c in cells:
            why_unusable = self._unusable_reason(c.strategy)
            if why_unusable:
                self._skip(c, f"strategy unavailable: {why_unusable}")
                continue
            why = D.skip_reason(c, self.st, self.cost, self.max_memory_gb, self.blocked)
            if why:
                self._skip(c, why)
            else:
                runnable.append(c)
        spent = 0.0
        for dev, variant, group in self._group(runnable, rep):
            keep, est = [], D.PRIORS["block_overhead_s"] + (D.PRIORS["mp_startup_s"] if variant.startswith("mp") else 0.0)
            for c in group:
                e = self._predicted(c)
                if spent + est + e > cap_s or est + e > self.remaining():
                    self._skip(c, f"{phase} time budget exhausted")
                else:
                    keep.append(c)
                    est += e
            if not keep:
                continue
            spent += est
            self._run_block(dev, variant, keep, phase, rep)
            self.cost.update(self.results, self.st)

    def _run_block(self, dev: str, variant: str, cells: List[Cell], phase: str, rep: int, retry: bool = False) -> None:
        self._n_blocks += 1
        spec = BlockSpec(f"{phase}-r{rep}-{dev}-{variant}-{self._n_blocks}", dev, variant, cells, phase, rep)
        self.log(f"  block {spec.block_id}: {len(cells)} cell(s) | elapsed {self.elapsed():.0f}s / {self.st.max_total_s:.0f}s")
        t = self.clock()
        res = self.runner(spec, max(self.remaining(), 30.0))
        returned = res.get("cells", [])
        for r in returned:
            r.setdefault("phase", phase)
            self.results.append(r)
        status = res.get("status", "completed")
        self.blocks.append({"block_id": spec.block_id, "device": dev, "variant": variant, "phase": phase, "rep": rep,
                            "status": status, "reason": res.get("reason", ""), "placement": res.get("placement"),
                            "n_planned": len(cells), "n_returned": len(returned), "wall_s": self.clock() - t,
                            "cold_start_s": (res.get("data") or {}).get("block", {}).get("cold_start_s"),
                            "attempts": res.get("attempts", 1)})
        self.notes.extend(n for n in res.get("notes", []) if n not in self.notes)
        name = Strategy(dev, "mp" if variant.startswith("mp") else variant, int(variant[2:]) if variant.startswith("mp") else 1).name
        for r in returned:                                   # remember the cheapest aborted configuration -> prune above it
            if r.get("status") == "aborted":
                p = Point(**r["point"])
                cost = self._predicted(Cell(p, Strategy(dev, r["strategy"]["mode"], r["strategy"].get("workers", 1))))
                key = (name, p.tier)
                self.blocked[key] = min(cost, self.blocked.get(key, float("inf")))
        if status in HARD_FAIL:
            self._mark_unusable(dev, name, status, res.get("reason", ""))
        if status in ("crashed", "killed", "aborted"):
            self.notes.append(f"{spec.block_id} ended '{status}': {res.get('reason', '')[:160]}")
        done_ids = {r["cell_id"] for r in returned}
        left = [c for c in cells if c.cell_id not in done_ids]
        if not left:
            return
        if self._unusable_reason(Strategy(dev, "mp" if variant.startswith("mp") else variant, int(variant[2:]) if variant.startswith("mp") else 1)):
            for c in left:
                self.results.append(self._incomplete(c, "unsupported" if status == "unsupported" else "failed",
                                                     self._unusable_reason(c.strategy) or ""))
            return
        left = [c for c in left if D.skip_reason(c, self.st, self.cost, self.max_memory_gb, self.blocked) is None]
        for c in cells:                                      # cells dropped by pruning are recorded, never silently lost
            if c.cell_id not in done_ids and c not in left:
                self._skip(c, "pruned after an equal or cheaper configuration of this strategy/tier aborted")
        if left and not retry and status in ("crashed", "killed", "aborted") and self.remaining() > 60:
            self.log(f"  re-queueing {len(left)} unfinished cell(s) once in a fresh process")
            self._run_block(dev, variant, left, phase, rep, retry=True)
        else:
            for c in left:
                self.results.append(self._incomplete(c, "incomplete", f"block ended '{status}' before this cell ran"))

    @staticmethod
    def _incomplete(c: Cell, status: str, reason: str) -> Dict[str, Any]:
        return {"cell_id": c.cell_id, "point": c.point.as_dict(), "strategy": {"device": c.strategy.device, "mode": c.strategy.mode, "workers": c.strategy.workers},
                "phase": c.phase, "rep": c.rep, "status": status, "reason": reason, "placement": None, "ips": None, "evaluations": 0,
                "comps": {}}

    # ------------------------------------------------------------------------------------ phases
    def run(self) -> Dict[str, Any]:
        try:
            self._phase("pilot", 0.40)
            self._phase("expansion", 0.30)
            self._phase("confirmation", 1.0)
            self.status = "completed"
        except BaseException as e:                         # keep partial results for the report, then re-raise
            self.status, self.reason = "interrupted", f"{type(e).__name__}: {e}"
            raise
        finally:
            self.phase_log["total"] = {"elapsed_s": self.elapsed()}
        return self.result()

    def _phase(self, name: str, frac: float) -> None:
        t = self.clock()
        n0 = len(self.results)
        cap = max(0.0, min(self.remaining(), frac * self.st.max_total_s if name != "confirmation" else self.remaining()))
        if name == "pilot":
            self.run_cells(D.pilot_cells(self.st), "pilot", 0, cap)
        elif name == "expansion":
            table = D.seq_ratio_table(self.results)
            cands = D.choose_candidates(table, 1 if self.st.quick else 2) or self._cpu_only_candidates()
            self.phase_log["candidates"] = [c["key"] for c in cands]
            cells, skipped = D.plan_expansion(cands, self.st, self.results, self.cost, cap, self.max_memory_gb, self.blocked,
                                              is_usable=lambda s: self._unusable_reason(s) is None)
            self.skipped.extend({**s, "phase": "expansion"} for s in skipped)
            self.run_cells(cells, "expansion", 0, cap)
        else:
            table = D.seq_ratio_table(self.results)
            plan = D.plan_confirmation(table, self.results, self.st) or self._cpu_only_confirmation()
            self.phase_log["confirmation_plan"] = [(p.key, [s.name for s in ss]) for p, ss in plan]
            for rep, cells in enumerate(D.confirmation_cells(plan, self.st)):
                if self.remaining() < 60:
                    for c in cells:
                        self._skip(c, "confirmation time budget exhausted (repetition not run)")
                    continue
                self.run_cells(cells, "confirmation", rep, self.remaining())
        self.phase_log[name] = {"wall_s": self.clock() - t, "cap_s": cap, "cells_returned": len(self.results) - n0}

    def _cpu_only_candidates(self) -> List[Dict[str, Any]]:
        """No usable paired GPU data (e.g. no GPU): still explore CPU strategies at the most informative CPU points."""
        cpu = [r for r in self.results if r["phase"] == "pilot" and r.get("status") == "completed" and r["strategy"]["device"] == "cpu"]
        if not cpu:
            return []
        top = max(cpu, key=lambda r: r["wall_s"])
        return [{"point": top["point"], "key": Point(**top["point"]).key, "ratio": 0.0}]

    def _cpu_only_confirmation(self) -> List[Tuple[Point, List[Strategy]]]:
        gain = D.best_strategy_gain(self.results)
        return [gain] if gain else []

    def result(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "cells": self.results, "blocks": self.blocks,
                "skipped": self.skipped, "notes": self.notes, "phase_log": self.phase_log, "unusable_strategies": self.unusable,
                "blocked": {f"{k[0]}|{k[1]}": v for k, v in self.blocked.items()}, "elapsed_s": self.elapsed(),
                "cost_model": {f"{k[0]}|{k[1]}": v for k, v in self.cost.fitted.items()}}
