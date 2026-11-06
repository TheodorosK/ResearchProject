import json
import math
import os
import tempfile
import unittest
from unittest import mock

from tfbench import cli, common, report, tfenv, workloads


class StatsTests(unittest.TestCase):
    def test_timing_stats(self):
        s = common.timing_stats([1.0, 2.0, 3.0, 4.0, 10.0])
        self.assertEqual(s["median"], 3.0)
        self.assertAlmostEqual(s["mean"], 4.0)
        self.assertAlmostEqual(s["stdev"], math.sqrt(sum((x - 4) ** 2 for x in [1, 2, 3, 4, 10]) / 4))
        self.assertAlmostEqual(s["cv"], s["stdev"] / 4.0)
        self.assertEqual((s["min"], s["max"], s["n"]), (1.0, 10.0, 5))
        self.assertEqual(common.timing_stats([2.0])["stdev"], 0.0)
        self.assertEqual(common.timing_stats([]), {})

    def test_sanitize_is_strict_json(self):
        import numpy as np
        obj = {"a": float("nan"), "b": [float("inf"), np.float32(1.5), np.int64(3)], "c": np.arange(3), "d": object()}
        out = common.sanitize(obj)
        json.dumps(out, allow_nan=False)
        self.assertIsNone(out["a"])
        self.assertEqual(out["b"][1:], [1.5, 3])
        self.assertEqual(out["c"], [0, 1, 2])

    def test_recorder_kinds_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.json")
            r = common.Recorder("job", p)
            self.assertEqual(r.status, "running")        # a killed worker is never mistaken for 'completed'
            r.measure("m", float("nan"), "s", kind="calculated", device="cpu")
            with self.assertRaises(ValueError):
                r.measure("m", 1, kind="guess")
            r.checkpoint()
            with open(p) as fh:
                data = json.load(fh)
            self.assertIsNone(data["measurements"][0]["value"])
            self.assertEqual(data["measurements"][0]["context"], {"device": "cpu"})


class MemoryClassificationTests(unittest.TestCase):
    def noisy(self, f, n=30):
        import random
        rng = random.Random(0)
        return [f(i) + rng.uniform(-0.2, 0.2) for i in range(n)]

    def test_flat_after_warmup_is_bounded(self):
        s = self.noisy(lambda i: 400 + 60 * (1 - math.exp(-i / 1.0)))
        self.assertEqual(common.classify_memory_growth(s)["verdict"], "bounded")

    def test_decelerating_growth_is_saturating(self):
        s = self.noisy(lambda i: 400 + 120 * (1 - math.exp(-i / 8.0)))
        self.assertEqual(common.classify_memory_growth(s, slope_threshold_mb=0.3)["verdict"], "saturating")

    def test_linear_native_growth_flagged_as_potential(self):
        s = self.noisy(lambda i: 400 + 4.0 * i)
        py = self.noisy(lambda i: 20.0)
        v = common.classify_memory_growth(s, py)
        self.assertEqual(v["verdict"], "continued_native_growth")
        self.assertAlmostEqual(v["slope_mb_per_cycle"], 4.0, delta=0.3)

    def test_python_level_retention(self):
        s = self.noisy(lambda i: 400 + 4.0 * i)
        py = self.noisy(lambda i: 20 + 3.5 * i)
        self.assertEqual(common.classify_memory_growth(s, py)["verdict"], "python_retention")

    def test_too_few_cycles_is_inconclusive(self):
        self.assertEqual(common.classify_memory_growth([1, 2, 3, 4, 5, 6, 7])["verdict"], "inconclusive")

    def test_linear_slope(self):
        self.assertAlmostEqual(common.linear_slope([1, 3, 5, 7]), 2.0)
        self.assertEqual(common.linear_slope([5]), 0.0)


class WorkloadTests(unittest.TestCase):
    def test_param_counts(self):
        self.assertEqual(workloads.mlp_params(10, 20, 2, 3), (10 * 20 + 20) + (20 * 20 + 20) + (20 * 3 + 3))
        self.assertEqual(workloads.convnet_params(), 896 + 18496 + 4096 * 128 + 128 + 1290)
        wl = workloads.perf_workload("mlp", 784, 1024, 3, 10)
        self.assertEqual(wl.train_flops_per_sample, 6 * wl.params)
        est = wl.estimate(128, 4)
        self.assertGreater(est["total_gb"], est["model_and_optimizer_gb"])
        with self.assertRaises(ValueError):
            workloads.perf_workload("nope", 1, 1, 1, 1)


class GlorotInitTests(unittest.TestCase):
    def test_deterministic_bounded_and_shaped(self):
        import numpy as np
        shapes = [(10, 20), (20,), (3, 3, 4, 8), (8,)]
        a, b = workloads.glorot_init(shapes, 5), workloads.glorot_init(shapes, 5)
        c = workloads.glorot_init(shapes, 6)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertFalse(np.array_equal(a[0], c[0]))
        self.assertEqual([x.shape for x in a], shapes)
        self.assertTrue((a[1] == 0).all() and (a[3] == 0).all())                 # biases zero
        self.assertTrue(np.abs(a[0]).max() <= (6 / 30) ** 0.5 + 1e-6)             # dense limit
        self.assertTrue(np.abs(a[2]).max() <= (6 / (36 + 72)) ** 0.5 + 1e-6)      # conv fans use receptive field
        self.assertEqual(a[0].dtype, np.float32)


class NumericGateTests(unittest.TestCase):
    def test_gate(self):
        ok = {"host_truth_loss": 2.0, "eager_forward_loss": 2.0000001, "graph_forward_loss": 1.9999999}
        self.assertIsNone(workloads.numeric_gate(ok))
        bad = dict(ok, graph_forward_loss=1.4)
        msg = workloads.numeric_gate(bad)
        self.assertIn("graph mode", msg)
        self.assertNotIn("eager mode", msg)
        self.assertIn("FAILED", msg)
        self.assertIn("eager mode", workloads.numeric_gate(dict(ok, eager_forward_loss=3.0)))
        self.assertIsNone(workloads.numeric_gate({"host_truth_loss": None, "graph_forward_loss": 9}))   # no truth: skipped
        self.assertIsNone(workloads.numeric_gate(None))
        self.assertIn("n/a", workloads.numeric_gate(dict(ok, graph_forward_loss=float("nan"))))
        self.assertIsNone(workloads.numeric_gate(dict(ok, graph_forward_loss=2.01), tol=0.01))


class PlacementErrorTests(unittest.TestCase):
    def test_detection(self):
        self.assertTrue(tfenv.is_placement_error(RuntimeError(
            "Could not satisfy explicit device specification '/device:GPU:0' because no supported kernel")))
        self.assertTrue(tfenv.is_placement_error(ValueError("Cannot assign a device for operation X")))
        self.assertFalse(tfenv.is_placement_error(ValueError("shapes (3,) and (4,) are incompatible")))
        self.assertFalse(tfenv.is_placement_error(MemoryError("out of memory")))


def args_for(*argv):
    return cli.parse_args(list(argv))


class CliTests(unittest.TestCase):
    def test_default_limits_match_spec(self):
        a = args_for()
        lim = cli.limits_from(a)
        self.assertLessEqual(lim.max_extra_gb, 6.0)
        self.assertEqual((lim.min_avail_gb, lim.max_sys_pct, lim.max_swap_mb, lim.max_runtime_s),
                         (2.0, 80.0, 512.0, 1200.0))
        self.assertEqual(a.tests, ["alloc", "train", "leak"])        # backwards-compatible default

    def test_grappler_and_gate_flags(self):
        a = args_for()
        self.assertEqual((a.grappler, a.skip_numeric_gate), ("default", False))
        a = args_for("--grappler", "no-remapper", "--skip-numeric-gate")
        self.assertEqual((a.grappler, a.skip_numeric_gate), ("no-remapper", True))

    def test_default_activation_space_excludes_elu_but_is_overridable(self):
        self.assertNotIn("elu", args_for().ne_activations)
        self.assertIn("elu", args_for("--ne-activations", "relu", "elu").ne_activations)

    def test_quick_defaults_but_explicit_flags_win(self):
        a = args_for("--quick", "--ne-pop", "9")
        self.assertEqual(a.ne_pop, 9)
        self.assertEqual(a.ne_generations, cli.QUICK["ne_generations"])
        self.assertLess(a.perf_steps, args_for().perf_steps)

    def test_legacy_swap_gb_alias(self):
        self.assertEqual(args_for("--max-swap-gb", "1").max_swap_mb, 1024)

    def test_validation_catches_bad_configs_without_raising_limits(self):
        a = args_for("--tests", "neuroevolution", "--ne-generations", "50")
        self.assertTrue(any("max-generations" in e for e in cli.validate_args(a)))
        self.assertEqual(a.max_generations, 10)                       # not auto-raised
        a = args_for("--tests", "neuroevolution", "--ne-pop", "4", "--ne-elite", "4")
        self.assertTrue(any("elite" in e for e in cli.validate_args(a)))
        a = args_for("--tests", "performance", "--perf-repeats", "2")
        self.assertTrue(cli.validate_args(a))
        a = args_for("--tests", "memdiag", "--memdiag-cycles", "5")
        self.assertTrue(cli.validate_args(a))
        self.assertEqual(cli.validate_args(args_for("--tests", "performance", "neuroevolution", "--quick")), [])

    def test_job_names_roundtrip_and_interleaving(self):
        for j in (cli.Job("alloc"), cli.Job("performance", "gpu", "mlp"), cli.Job("neuroevolution", "cpu"),
                  cli.Job("memdiag", "gpu", "clear")):
            self.assertEqual(cli.Job.parse(j.name), j)
        a = args_for("--tests", "performance")
        names = [j.name for j in cli.plan_jobs(a)[0][1]]
        self.assertEqual(names, ["performance@cpu:mlp", "performance@gpu:mlp",
                                 "performance@cpu:convnet", "performance@gpu:convnet"])

    def test_dry_run_trains_nothing_and_reports(self):
        import io
        from contextlib import redirect_stdout
        a = args_for("--tests", "performance", "neuroevolution", "--quick", "--dry-run")
        buf = io.StringIO()
        with redirect_stdout(buf), mock.patch.object(cli.subprocess, "Popen") as popen:
            rc = cli.dry_run(a, cli.limits_from(a))
        popen.assert_not_called()
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("ESTIMATES", out)
        self.assertIn("evaluations per device", out)

    def test_dry_run_rejects_workload_that_exceeds_limit(self):
        import io
        from contextlib import redirect_stdout
        a = args_for("--tests", "performance", "--perf-width", "8192", "--perf-depth", "8", "--max-gb", "1")
        with redirect_stdout(io.StringIO()):
            rc = cli.dry_run(a, cli.limits_from(a))
        self.assertEqual(rc, 2)
        self.assertEqual(a.max_gb, 1.0)

    def test_evaluation_count_formula(self):
        a = args_for("--tests", "neuroevolution", "--ne-pop", "10", "--ne-elite", "2", "--ne-generations", "4")
        self.assertEqual(cli.estimate_plan(a)["neuroevolution"]["evaluations_per_device"], 10 + 8 * 3)


class SupervisorRetryTests(unittest.TestCase):
    @staticmethod
    def fake(status_by_placement):
        calls = []

        def run_job(args, limits, out_dir, job, placement, remaining):
            calls.append((job.name, placement))
            return {"job": job.name, "status": status_by_placement[placement], "reason": "no kernel for op X",
                    "measurements": [], "data": {}, "notes": [], "placement": placement}
        return run_job, calls

    def go(self, argv, status):
        a = args_for(*argv)
        fake, calls = self.fake(status)
        with mock.patch.object(cli, "run_job", fake):
            res, interrupted = cli.supervise(a, cli.limits_from(a), "/nonexistent")
        return res, calls

    def test_auto_retries_gpu_once_in_soft_mode_and_labels_it(self):
        res, calls = self.go(["--tests", "performance", "--perf-workloads", "mlp"],
                             {"strict": "placement_failed", "soft": "completed"})
        self.assertEqual(calls, [("performance@cpu:mlp", "strict"), ("performance@gpu:mlp", "strict"),
                                 ("performance@gpu:mlp", "soft")])
        gpu = res[1]
        self.assertEqual((gpu["status"], gpu["placement"], gpu["attempts"]), ("completed", "soft", 2))
        self.assertIn("no kernel", gpu["placement_strict_failure"])
        self.assertTrue(any("soft placement" in n for n in gpu["notes"]))

    def test_strict_mode_never_retries(self):
        res, calls = self.go(["--tests", "performance", "--perf-workloads", "mlp", "--placement", "strict"],
                             {"strict": "placement_failed", "soft": "completed"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(res[1]["status"], "placement_failed")

    def test_soft_mode_runs_soft_directly(self):
        _, calls = self.go(["--tests", "performance", "--perf-workloads", "mlp", "--placement", "soft"],
                           {"soft": "completed", "strict": "completed"})
        self.assertEqual(calls, [("performance@cpu:mlp", "soft"), ("performance@gpu:mlp", "soft")])

    def test_cpu_failures_are_never_retried(self):
        _, calls = self.go(["--tests", "neuroevolution", "--ne-devices", "cpu"],
                           {"strict": "placement_failed", "soft": "completed"})
        self.assertEqual(calls, [("neuroevolution@cpu", "strict")])


def fixture_job(device, step_ms, loss=2.30, status="completed", workload="mlp"):
    ctx = dict(device=device, workload=workload)
    m = lambda n, v, k="measured": {"name": n, "value": v, "unit": "", "kind": k, "context": ctx}
    meas = [] if status != "completed" else [
        m("steady_step_time_ms", step_ms, "calculated"), m("steady_samples_per_s", 256000 / step_ms, "calculated"),
        m("steady_steps_per_s", 1000 / step_ms, "calculated"), m("loss_first_step", loss),
        m("peak_extra_system_gb_sampled", 0.5), m("peak_rss_gb", 0.6), m("min_available_gb_sampled", 8.0),
        m("max_system_pct_sampled", 55.0), m("swap_growth_during_steady_mb", 0.0)]
    return {"job": f"performance@{device}:{workload}", "test": "performance", "device": device, "workload": workload,
            "placement": "strict", "status": status, "reason": "" if status == "completed" else "boom",
            "measurements": meas, "data": {"device_evidence": {"verified": True}}, "notes": []}


class ReportTests(unittest.TestCase):
    def build(self, jobs):
        meta = {"run_id": "t", "started_utc": "s", "finished_utc": "f", "command": "c",
                "hardware": {"chip": "Apple M1 Pro"}, "software": {"tensorflow": "2.15.1"},
                "limits": {"max_extra_gb": 6}, "config": {"seed": 1}}
        return report.build_report(meta, jobs)

    def test_speedup_is_calculated_only_from_two_completed_jobs(self):
        r = self.build([fixture_job("cpu", 90.0), fixture_job("gpu", 9.0)])
        row = r["derived"]["cpu_vs_gpu_performance"][0]
        self.assertAlmostEqual(row["gpu_speedup_x"], 10.0)
        self.assertEqual(row["kind"], "calculated")
        self.assertTrue(row["workloads_comparable"])
        r = self.build([fixture_job("cpu", 90.0), fixture_job("gpu", 0, status="unsupported")])
        self.assertEqual(r["derived"]["cpu_vs_gpu_performance"], [])
        self.assertEqual(r["incomplete"][0]["status"], "unsupported")

    def test_mismatching_first_step_loss_marks_workloads_not_comparable(self):
        r = self.build([fixture_job("cpu", 90.0, loss=2.30), fixture_job("gpu", 9.0, loss=2.90)])
        self.assertFalse(r["derived"]["cpu_vs_gpu_performance"][0]["workloads_comparable"])

    def test_outputs_are_valid_and_complete(self):
        r = self.build([fixture_job("cpu", 90.0), fixture_job("gpu", 9.0), fixture_job("gpu", 1, status="aborted", workload="convnet")])
        with tempfile.TemporaryDirectory() as d:
            paths = report.write_reports(r, d)
            with open(paths["json"]) as fh:
                loaded = json.load(fh, parse_constant=lambda c: self.fail(f"non-standard JSON {c}"))
            self.assertEqual(loaded["schema_version"], 1)
            with open(paths["csv"]) as fh:
                rows = fh.read().splitlines()
            self.assertEqual(rows[0].split(",")[:9], ["job", "test", "device", "workload", "placement", "name",
                                                       "value", "unit", "kind"])
            self.assertGreater(len(rows), 10)
            with open(paths["markdown"]) as fh:
                md = fh.read()
        for n, title in enumerate(["Hardware and software", "TensorFlow and device", "CPU and GPU performance",
                                   "Baseline and peak memory", "Swap activity", "Neuroevolution throughput",
                                   "Time per generation", "Memory growth", "Incomplete or aborted",
                                   "Measurement limitations"], start=1):
            self.assertRegex(md, rf"## {n}\. {title}")
        self.assertIn("performance@gpu:convnet", md)             # aborted job listed in section 9
        self.assertIn("No performance claim is made about other Apple Silicon", md)
        self.assertNotIn("M2", md)
        self.assertNotIn("M3", md)

    def test_neuroevolution_numerical_agreement_flag(self):
        def ne(device, r2):
            m = lambda n, v: {"name": n, "value": v, "unit": "", "kind": "measured", "context": {"device": device}}
            return {"job": f"neuroevolution@{device}", "test": "neuroevolution", "device": device, "workload": None,
                    "placement": "strict", "status": "completed", "reason": "", "notes": [], "data": {},
                    "measurements": [m("overall_individuals_per_s", 5.0), m("reference_val_r2", r2)]}
        ok = self.build([ne("cpu", 0.60), ne("gpu", 0.62)])["derived"]["cpu_vs_gpu_neuroevolution"]
        bad = self.build([ne("cpu", 0.60), ne("gpu", 0.10)])["derived"]["cpu_vs_gpu_neuroevolution"]
        self.assertTrue(ok["numerically_consistent"])
        self.assertFalse(bad["numerically_consistent"])
        md = report.render_markdown(self.build([ne("cpu", 0.60), ne("gpu", 0.10)]))
        self.assertIn("NOT consistent", md)

    def test_first_divergent_step_is_reported(self):
        def ne(device, trace):
            m = lambda n, v: {"name": n, "value": v, "unit": "", "kind": "measured", "context": {"device": device}}
            return {"job": f"neuroevolution@{device}", "test": "neuroevolution", "device": device, "workload": None,
                    "placement": "strict", "status": "completed", "reason": "", "notes": [],
                    "data": {"reference_check": {"loss_trace": trace}},
                    "measurements": [m("overall_individuals_per_s", 5.0), m("reference_val_r2", 0.5)]}
        same = self.build([ne("cpu", [1.0, 0.9, 0.8]), ne("gpu", [1.0, 0.9, 0.8])])["derived"]["cpu_vs_gpu_neuroevolution"]
        self.assertIsNone(same["first_divergent_step_rel_1e-3"])
        div = self.build([ne("cpu", [1.0, 0.9, 0.8]), ne("gpu", [1.0, 0.9, 1.5])])
        self.assertEqual(div["derived"]["cpu_vs_gpu_neuroevolution"]["first_divergent_step_rel_1e-3"], 2)
        self.assertIn("First step where CPU and GPU differ", report.render_markdown(div))

    def test_kind_legend_present(self):
        r = self.build([])
        self.assertEqual(set(r["kind_legend"]), {"measured", "calculated", "estimated"})


if __name__ == "__main__":
    unittest.main()
