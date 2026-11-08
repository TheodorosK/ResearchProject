import csv
import importlib.util
import json
import os
import tempfile
import unittest

from tfbench.scaling import lifecycle as LC
from tfbench.scaling.controller import BlockSpec
from tfbench.scaling.workload import Settings

spec = importlib.util.spec_from_file_location("memwatch", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "memwatch.py"))
mw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mw)

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               81920.
Pages active:                            262144.
Pages inactive:                          196608.
Pages speculative:                         4096.
Pages wired down:                        131072.
Pages occupied by compressor:            163840.
"""
FOOTPRINT = """foo [123]: 64-bit    Footprint: 1536 MB (16384 bytes per page)

    Dirty      Clean  Reclaimable    Regions    Category
      ---        ---          ---        ---    ---
  1024 MB        0 B          0 B         40    IOAccelerator
   300 MB       10 MB          0 B        120    MALLOC_LARGE
"""
GB = 1024 ** 3


def st(**kw):
    base = dict(design="lifecycle", steps=(20,), generations=2, lifecycle_pops=(4, 8, 16), lifecycle_rounds=3, max_total_s=1200.0)
    base.update(kw)
    return Settings(**base)


class TestParsers(unittest.TestCase):
    def test_vm_stat(self):
        d = mw.parse_vm_stat(VM_STAT)
        self.assertAlmostEqual(d["wired"], 131072 * 16384 / GB)
        self.assertAlmostEqual(d["compressed"], 163840 * 16384 / GB)
        self.assertEqual(mw.parse_vm_stat("garbage"), {})

    def test_memory_pressure(self):
        self.assertEqual(mw.parse_memory_pressure("System-wide memory free percentage: 41%"), 41.0)
        self.assertIsNone(mw.parse_memory_pressure(""))

    def test_footprint(self):
        d = LC.parse_footprint(FOOTPRINT)
        self.assertAlmostEqual(d["footprint_gb"], 1.5)
        self.assertAlmostEqual(d["categories"]["IOAccelerator"]["dirty_gb"], 1.0)
        self.assertEqual(LC.parse_footprint("nothing")["footprint_gb"], None)


def snap(avail, rss, uss=None):
    return {"label": "x", "t": 0, "avail_gb": avail, "sys_pct": 50, "swap_used_gb": 0.0, "swap_out_gb": 0.0, "rss_gb": rss, "uss_gb": uss if uss is not None else rss,
            "footprint_gb": None, "footprint_categories": None}


def long_data(increments, pops=(4, 8, 16)):
    """Synthetic long-lived record: ``increments`` = extra GB held after each round (system-wide)."""
    ready = snap(10.0, 0.5)
    cells, held = [], 0.0
    for r, inc in enumerate(increments):
        for i, p in enumerate(pops):
            before = held
            held += inc / len(pops)
            s = lambda a: snap(10.0 - a, 0.5 + a)
            cells.append({"cell_id": f"c{r}{p}", "population": p, "rep": r, "status": "completed", "gc_s": 0.1, "ips": 2.0, "wall_s": 5.0,
                          "peak_additional_gb": 1.0, "peak_extra_block_gb": held + 1.0, "min_avail_gb": 5.0,
                          "snaps": {"before": s(before), "after_exec": s(held), "after_gc": s(held), "settled": s(held)}})
    return {"label": "long", "runtime_ready": ready, "cells": cells, "stopped": None, "headroom_gb": 6.0}


class TestRetention(unittest.TestCase):
    def verdict(self, incs):
        rows = LC.long_rows(long_data(incs))
        return LC.classify_retention(rows, len(incs), 3)

    def test_plateau_is_stabilised_not_a_leak(self):
        v = self.verdict([1.2, 0.05, 0.02])
        self.assertEqual(v["verdict"], "stabilised")

    def test_steady_growth_is_flagged_as_still_growing(self):
        v = self.verdict([1.0, 1.0, 0.9])
        self.assertEqual(v["verdict"], "still_growing")

    def test_decaying_growth_is_partly_reused(self):
        self.assertEqual(self.verdict([1.0, 0.5, 0.3])["verdict"], "partly_reused")

    def test_fewer_than_three_rounds_is_inconclusive(self):
        self.assertEqual(self.verdict([1.0, 0.0])["verdict"], "inconclusive")

    def test_markdown_avoids_leak_claims_without_evidence(self):
        an = LC.analyze_lifecycle({"lifecycle_long": long_data([1.2, 0.05, 0.02]), "lifecycle_fresh": [], "lifecycle_variants": []}, st(), 6.0)
        md = LC.markdown(an, st(), "completed", 300.0, [])
        self.assertIn("stabilised", md)
        self.assertIn("not, by themselves, evidence of a leak", md)


class FakeRunner:
    def __init__(self, settings, long_incs=(1.0, 0.0, 0.0)):
        self.st, self.long_incs, self.calls = settings, long_incs, []

    def __call__(self, block: BlockSpec, rem):
        self.calls.append((block.variant, len(block.cells), block.extra.get("label"), block.extra.get("mode")))
        if block.variant == "lifecycle" and block.extra.get("label") == "long":
            return {"status": "completed", "cells": [], "notes": [], "data": {"lifecycle": long_data(self.long_incs)}}
        if block.variant == "lifecycle":
            d = long_data([0.0])
            d["cells"] = d["cells"][:1]
            d["runtime_ready"] = snap(9.0, 0.5)
            return {"status": "completed", "cells": [], "notes": [], "data": {"lifecycle": d}}
        rungs = [{"n_resident": n, "total_params": n * 100000, "extra_gb": 0.1 + 0.08 * n, "theory_gb": 0.0, "uss_gb": 0.05 * n, "footprint_gb": None,
                  "build_s_per_model": 0.1, "step_ms_per_model": None, "peak_extra_gb": None, "rss_gb": 1, "avail_gb": 9, "min_avail_gb": 9, "swap_growth_mb": 0, "projected_gb": 0}
                 for n in (1, 2, 4, 8)]
        return {"status": "completed", "cells": [], "notes": [], "data": {"resident": {"tier": "medium", "device": "gpu", "mode": block.extra["mode"], "homo": block.extra["homo"],
                                                                                    "rungs": rungs, "stopped": None, "release": None}}}


class TestController(unittest.TestCase):
    def run_ctrl(self, settings, headroom=6.0):
        r = FakeRunner(settings)
        c = LC.LifecycleController(settings, headroom, r, log=lambda *_: None)
        c.run()
        return c, r

    def test_sequence_long_then_fresh_then_medium_variants(self):
        c, r = self.run_ctrl(st())
        kinds = [(k[0], k[2]) for k in r.calls]
        self.assertEqual(kinds[0], ("lifecycle", "long"))
        self.assertEqual(sum(1 for k in kinds if k == ("lifecycle", "fresh")), 9)
        self.assertEqual({k[3] for k in r.calls if k[0] == "resident"}, {"full", "trace", "vars"})
        self.assertEqual(len(c.fresh), 9)
        self.assertTrue(all("supervisor" in f for f in c.fresh))
        an = LC.analyze_lifecycle(c.result(), st(), 6.0)
        self.assertEqual(len(an["variants"]), 4)
        self.assertEqual(an["retention"]["verdict"], "stabilised")
        self.assertIn("Fresh process per cell", LC.markdown(an, st(), "completed", 600.0, c.skipped))

    def test_skip_flags(self):
        c, r = self.run_ctrl(st(lifecycle_fresh=False, lifecycle_medium=False))
        self.assertEqual(len(r.calls), 1)

    def test_cells_that_do_not_fit_are_skipped_with_reason(self):
        c, r = self.run_ctrl(st(), headroom=2.5)         # P16 estimate x1.5 > 2.5
        self.assertTrue(any("P16" in s["cell_id"] and "estimated memory" in s["reason"] for s in c.skipped))

    def test_design_summary(self):
        d = LC.design_summary_lifecycle(st(), 6.0)
        self.assertEqual((d["cells"], d["blocks"]), (9, 1 + 9 + 4))


class TestMemwatch(unittest.TestCase):
    def write_csv(self, d, name, scenario, demand, swap=0.0, pressure=1):
        path = os.path.join(d, name + ".csv")
        sampler_cols = mw.BASE_COLS
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sampler_cols)
            w.writeheader()
            for i, dem in enumerate(demand):
                w.writerow({"t_iso": "t", "elapsed_s": i * 5, "avail_gb": 16 - dem, "used_pct": 100 * dem / 16, "demand_gb": dem, "compressed_gb": 1.0,
                            "swap_used_gb": swap, "swap_growth_mb": 1024 * swap, "swapin_mb_s": 0, "swapout_mb_s": 0, "pressure_level": pressure,
                            "cpu_pct": 20, "sched_lag_ms": 1.0, "spawn_ms": 5.0, "marker": "go" if i == 3 else ""})
        with open(path.replace(".csv", ".meta.json"), "w") as fh:
            json.dump({"scenario": scenario, "note": "", "total_gb": 16.0}, fh)
        return path

    def test_summary_and_decomposition(self):
        with tempfile.TemporaryDirectory() as d:
            a = mw.summarize_csv(self.write_csv(d, "a", "A", [8.0] * 30))
            b = mw.summarize_csv(self.write_csv(d, "b", "B", [8.0] * 20 + [8.8] * 20))
            c = mw.summarize_csv(self.write_csv(d, "c", "C", [8.0] * 20 + [12.0] * 20, swap=0.3, pressure=2))
            self.assertAlmostEqual(a["demand_start_gb"], 8.0)
            self.assertAlmostEqual(b["demand_increase_max_gb"], 0.8, places=5)
            self.assertEqual(c["pressure_share_not_normal"], 1.0)
            self.assertEqual(a["markers"][0]["text"], "go")
            out = mw.compare([a, b, c])
            self.assertIn("Decomposition", out)
            self.assertIn("interference", out)
            self.assertIn("A 16 GB machine is sufficient for scenario C only if", out)

    def test_record_runs_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            rc = mw.main(["record", "--scenario", "T", "--interval", "1", "--max-minutes", "0.04", "--out-dir", d, "--track", "python"])
            self.assertEqual(rc, 0)
            files = os.listdir(d)
            for ext in (".csv", ".summary.json", ".meta.json", ".procs.jsonl"):
                self.assertTrue(any(f.endswith(ext) for f in files), ext)

    def test_interval_below_one_second_refused(self):
        self.assertEqual(mw.main(["record", "--scenario", "T", "--interval", "0.2"]), 2)


if __name__ == "__main__":
    unittest.main()
