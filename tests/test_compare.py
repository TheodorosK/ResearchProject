import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from tests.test_common_report_cli import fixture_job
from tfbench import report

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compare_reports", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "compare_reports.py"))
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
except Exception as e:                                            # pragma: no cover
    compare = None


def make_report(step_ms, tf="2.15.1", grappler="none", chip="Apple M1 Pro"):
    jobs = [fixture_job("cpu", 90.0), fixture_job("gpu", step_ms)]
    meta = {"run_id": "r", "started_utc": "s", "finished_utc": "f", "command": "c",
            "hardware": {"chip": chip, "ram_total_gb": 16.0}, "software": {"tensorflow": tf, "tensorflow-metal": "1.2.0"},
            "limits": {}, "config": {"seed": 1234, "grappler": grappler, "quick": False}}
    return report.build_report(meta, jobs)


@unittest.skipIf(compare is None, "compare tool not importable")
class CompareTests(unittest.TestCase):
    def run_compare(self, *reports):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, r in enumerate(reports):
                p = os.path.join(d, f"r{i}.json")
                with open(p, "w") as fh:
                    json.dump(r, fh)
                paths.append(p)
            buf = io.StringIO()
            with redirect_stdout(buf):
                compare.main(paths)
        return buf.getvalue()

    def test_like_for_like_with_ratio_and_verdicts(self):
        out = self.run_compare(make_report(9.0), make_report(9.1), make_report(6.0))
        self.assertIn("like-for-like", out)
        self.assertIn("similar", out)             # 9.1 vs 9.0 ms
        self.assertIn("better", out)              # 6.0 ms is faster than 9.0 ms
        self.assertIn("performance@gpu:mlp", out)

    def test_mismatch_is_flagged_and_ratios_marked_meaningless(self):
        out = self.run_compare(make_report(9.0), make_report(9.0, tf="2.16.0", grappler="default", chip="Apple M4 Max"))
        self.assertIn("WARNING", out)
        self.assertIn("tensorflow", out)
        self.assertIn("grappler", out)
        self.assertIn("NOT meaningful", out)
        self.assertIn("hardware differs", out)

    def test_similar_band_is_configurable(self):
        a, b = make_report(9.0), make_report(9.5)               # 5.6 % slower
        self.assertIn("similar", self.run_compare(a, b))        # inside the default 7 % band
        with tempfile.TemporaryDirectory() as d:
            ps = []
            for i, r in enumerate((a, b)):
                ps.append(os.path.join(d, f"r{i}.json"))
                with open(ps[-1], "w") as fh:
                    json.dump(r, fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                compare.main([*ps, "--similar", "0.02"])
        self.assertIn("worse", buf.getvalue())

    def test_volatile_metrics_get_a_wide_band(self):
        self.assertEqual(compare.verdict(0.5, False, compare.VOLATILE_BAND), "similar")     # e.g. memory peak halved: noise
        self.assertEqual(compare.verdict(0.5, False, compare.SIMILAR), "better")
        self.assertIn("first_execution_s", compare.VOLATILE)
        self.assertNotIn("steady_samples_per_s", compare.VOLATILE)

    def test_needs_two_reports(self):
        with self.assertRaises(SystemExit):
            compare.main(["only_one.json"])


if __name__ == "__main__":
    unittest.main()
