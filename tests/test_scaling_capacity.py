import unittest

import numpy as np

from tfbench.scaling import capacity as C
from tfbench.scaling.controller import BlockSpec
from tfbench.scaling.workload import Cell, Point, Settings, Strategy

GB = 1024 ** 3


def settings(**kw):
    base = dict(design="capacity", tiers=("large",), pops=(4, 8, 16, 32), steps=(20,), generations=2, workers=(2, 4), max_workers=4,
                ladder=(1, 2, 4, 8, 16, 32), resident_tiers=("large",), max_total_s=1200.0)
    base.update(kw)
    return Settings(**base)


def rungs(device, per_param, icpt, st, tier="large", counts=(1, 2, 4, 8)):
    from tfbench.scaling.workload import Point as P, initial_population, params_of
    ps = [params_of(g, st) for g in initial_population(P(tier, max(counts), 1, "hetero", 1), st)]
    out = []
    for n in counts:
        tot = sum(ps[:n])
        out.append({"n_resident": n, "total_params": tot, "extra_gb": icpt + per_param * tot, "peak_extra_gb": icpt + per_param * tot,
                    "rss_gb": 1.0, "avail_gb": 10.0, "min_avail_gb": 9.0, "swap_growth_mb": 0.0, "theory_gb": tot * 16 / GB,
                    "build_s_added": 1.0 * n, "build_s_per_model": 0.9, "step_ms_per_model": 9.5, "projected_gb": 0.0})
    return {"tier": tier, "device": device, "ladder": list(counts), "headroom_gb": 6.0, "rungs": out, "stopped": None, "kind": "measured",
            "release": {"n_released": counts[-1], "peak_extra_gb": out[-1]["extra_gb"], "extra_before_release_gb": out[-1]["extra_gb"],
                        "extra_after_dropping_refs_gb": out[-1]["extra_gb"], "extra_after_gc_gb": 0.5 * out[-1]["extra_gb"],
                        "extra_after_3s_gb": 0.4 * out[-1]["extra_gb"], "gc_s": 0.1, "returned_fraction": 0.6, "swap_growth_mb": 0.0}}


def cell_result(c: Cell, ips=2.0, peak=0.8):
    return {"cell_id": c.cell_id, "point": c.point.as_dict(), "strategy": {"device": c.strategy.device, "mode": c.strategy.mode, "workers": c.strategy.workers},
            "phase": "capacity", "rep": 0, "status": "completed", "reason": "", "placement": "strict", "ips": ips, "wall_s": 10.0,
            "evaluations": c.point.population * 2, "comps": {"construct": 1, "trace": 1, "first_exec": 2, "train": 5, "fitness": 0.1, "teardown": 0.5},
            "peak_additional_gb": peak, "retained_gb": 0.01, "retained_sys_gb": 0.02, "swap_growth_mb": 0.0, "min_avail_gb": 9.0, "evaluations_done": 1,
            "evaluations_x": 0}


class TestGrid(unittest.TestCase):
    def test_cells_cover_requested_axes_without_cartesian_blowup(self):
        g = C.capacity_cells(settings())
        self.assertEqual(len(g["batched"]), 8)          # 4 P x 2 devices, homogeneous
        self.assertEqual(len(g["seq"]), 8)
        self.assertEqual(len(g["mp"]), 8)               # cpu only, 2 worker counts
        self.assertTrue(all(c.point.mode == "homo" for c in g["batched"]))
        self.assertTrue(all(c.point.mode == "hetero" for c in g["seq"] + g["mp"]))
        self.assertTrue(all(c.strategy.device == "cpu" for c in g["mp"]))

    def test_skip_flags_and_worker_cap(self):
        g = C.capacity_cells(settings(skip_batched=True, skip_mp=True))
        self.assertEqual((len(g["batched"]), len(g["mp"])), (0, 0))
        g = C.capacity_cells(settings(workers=(2, 4), max_workers=2))
        self.assertEqual({c.strategy.workers for c in g["mp"]}, {2})

    def test_design_summary(self):
        d = C.design_summary_capacity(settings(), 6.0)
        self.assertEqual((d["grid_cells"], d["resident_blocks"]), (24, 2))
        self.assertGreater(d["estimated_total_s"], 0)
        d2 = C.design_summary_capacity(settings(), 1.0)       # tiny headroom -> estimate-based skips are reported, not errors
        self.assertTrue(d2["cells_over_headroom_by_estimate"])


class TestProjection(unittest.TestCase):
    def test_first_rungs_are_pessimistic_and_later_use_measured_slope(self):
        st = settings()
        first = C._project_gb([], 0.0, [4_000_000], st, "large", "gpu")
        self.assertGreater(first, 3 * 4_000_000 * 16 / GB)
        rs = rungs("gpu", 100 / GB, 0.9, st)["rungs"]
        proj = C._project_gb(rs, rs[-1]["extra_gb"], [4_000_000] * 8, st, "large", "gpu")
        self.assertGreater(proj, rs[-1]["extra_gb"] + 1.3 * 100 / GB * 8 * 4_000_000 * 0.99)

    def test_projection_never_below_theory(self):
        st = settings()
        rs = rungs("cpu", 0.0, 0.1, st)["rungs"]              # measured slope zero (noise) must not predict free models
        proj = C._project_gb(rs, 0.1, [4_000_000] * 4, st, "large", "cpu")
        self.assertGreaterEqual(proj, 0.1 + 4 * 4_000_000 * 16 / GB)


class TestFit(unittest.TestCase):
    def test_recovers_linear_ground_truth(self):
        st = settings()
        r = rungs("gpu", 120 / GB, 0.9, st)
        f = C.fit_resident(r)
        self.assertAlmostEqual(f["intercept_gb"], 0.9, places=6)
        self.assertAlmostEqual(f["bytes_per_param"], 120.0, places=3)
        self.assertGreater(f["overhead_factor"], 5)           # 120 measured vs ~16-19 theoretical
        self.assertGreater(f["r2"], 0.999999)

    def test_too_few_rungs_gives_no_fit(self):
        st = settings()
        r = rungs("gpu", 1e-9, 0.1, st, counts=(1, 2))
        self.assertNotIn("gb_per_param", C.fit_resident(r))

    def test_projection_table_is_labelled_calculated(self):
        st = settings()
        f = C.fit_resident(rungs("gpu", 120 / GB, 0.9, st))
        p = C.capacity_projection([f])[0]
        self.assertEqual(p["kind"], "calculated")
        self.assertLess(p["models_fitting"]["6"], p["models_fitting"]["48"])


class FakeRunner:
    def __init__(self, st, per_param=100 / GB, icpt=0.9):
        self.st, self.per_param, self.icpt, self.calls = st, per_param, icpt, []

    def __call__(self, block: BlockSpec, remaining_s):
        self.calls.append((block.device, block.variant, [c.cell_id for c in block.cells]))
        if block.variant == "resident":
            return {"status": "completed", "reason": "", "cells": [], "placement": "strict", "notes": [],
                    "data": {"resident": rungs(block.device, self.per_param, self.icpt, self.st, block.extra["tier"])}}
        return {"status": "completed", "reason": "", "cells": [cell_result(c) for c in block.cells], "placement": "strict", "notes": [], "data": {}}


class TestController(unittest.TestCase):
    def run_ctrl(self, st, headroom, **kw):
        runner = FakeRunner(st, **kw)
        ctrl = C.CapacityController(st, headroom, runner, log=lambda *_: None)
        ctrl.run()
        return ctrl, runner

    def test_probe_runs_first_then_grid_and_fills_every_cell_or_records_skip(self):
        st = settings()
        ctrl, runner = self.run_ctrl(st, 6.0)
        self.assertEqual(runner.calls[0][1], "resident")
        ids = {c["cell_id"] for c in ctrl.results} | {s["cell_id"] for s in ctrl.skipped}
        for grp in C.capacity_cells(st).values():
            for c in grp:
                self.assertIn(c.cell_id, ids, "every grid cell is either measured or skipped with a reason")
        self.assertTrue(all(s["reason"] for s in ctrl.skipped))
        self.assertEqual(len(ctrl.resident), 2)               # gpu + cpu probes

    def test_measured_slope_vetoes_batched_cells_that_would_not_fit(self):
        st = settings()
        ctrl, _ = self.run_ctrl(st, 6.0, per_param=150 / GB)       # ~8x theory: P=4 fits, P=32 cannot
        sk = {s["cell_id"]: s["reason"] for s in ctrl.skipped}
        big = [k for k in sk if "P32" in k and "batched" in k]
        self.assertTrue(big)
        self.assertTrue(all("memory:" in sk[k] or "estimated memory" in sk[k] for k in big))
        done = {c["cell_id"] for c in ctrl.results}
        self.assertTrue(any("P4/" in k and "gpu-batched" in k for k in done))      # small ones still run

    def test_sequential_cells_are_not_vetoed_by_population_size(self):
        st = settings()
        ctrl, _ = self.run_ctrl(st, 6.0, per_param=150 / GB)
        seq_done = [c for c in ctrl.results if c["strategy"]["mode"] == "seq"]
        self.assertEqual(len(seq_done), 8)                        # P up to 32 sequentially: one resident model at a time

    def test_unusable_gpu_probe_is_recorded_and_skips_gpu_cells(self):
        st = settings()

        def runner(block, rem):
            if block.device == "gpu":
                return {"status": "unsupported", "reason": "no GPU is visible", "cells": [], "placement": "strict", "notes": [], "data": {}}
            return FakeRunner(st)(block, rem)
        ctrl = C.CapacityController(st, 6.0, runner, log=lambda *_: None)
        ctrl.run()
        self.assertTrue(any(s["strategy"] == "gpu-resident" for s in ctrl.skipped))
        self.assertTrue(any("gpu-seq" in s["strategy"] or s["strategy"] == "gpu-seq" for s in ctrl.skipped))
        self.assertFalse([c for c in ctrl.results if c["strategy"]["device"] == "gpu"])

    def test_time_budget_exhaustion_is_recorded(self):
        st = settings(max_total_s=60.0)
        ctrl, runner = self.run_ctrl(st, 6.0)
        self.assertEqual(runner.calls, [])                        # < 120 s left: nothing started
        self.assertTrue(ctrl.skipped and all("budget" in s["reason"] for s in ctrl.skipped))


class TestAnalysis(unittest.TestCase):
    def test_analysis_and_markdown(self):
        st = settings()
        ctrl = C.CapacityController(st, 6.0, FakeRunner(st), log=lambda *_: None)
        ctrl.run()
        an = C.analyze_capacity(ctrl.result(), st, 6.0)
        self.assertTrue(an["table"] and an["resident_fits"] and an["projection_calculated"])
        self.assertIn("memory_capacity", an["readings"])
        batched = [r for r in an["table"] if r["strategy"] == "gpu-batched" and r["status"] == "completed"]
        self.assertTrue(batched and batched[0]["vs_seq_same_device"] is not None)
        md = C.markdown(an, st, "completed", 600.0, ctrl.skipped)
        for needle in ("Resident-model probe", "CALCULATED", "Skipped or incomplete", "3d. Construction and cleanup"):
            self.assertIn(needle, md)

    def test_resident_probe_skips_are_not_grid_rows(self):
        rows = C.capacity_table([], [{"strategy": "gpu-resident", "point": {"tier": "large", "population": 32, "steps": 1}, "reason": "x"}])
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
