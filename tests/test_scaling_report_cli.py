import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from xml.etree import ElementTree as ET

from tests.test_common_report_cli import args_for
from tfbench import cli, common, report as main_report
from tfbench.scaling import analysis as A, design as D, report as SR, run as R, scheduler as SC, synthetic as SY, workload as W

_CACHE = {}


def synth_job(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        st = kw.pop("st", None) or W.Settings()
        clock = [0.0]
        _CACHE[key] = (st, R.run_scaling(st, 6.0, SY.make_runner(st, clock, **kw), clock=lambda: clock[0], log=lambda *_: None))
    return _CACHE[key]


class CliTests(unittest.TestCase):
    def test_defaults_match_the_specification(self):
        a = args_for("--tests", "scaling")
        self.assertEqual((a.scaling_tiers, a.scaling_pops, a.scaling_steps), (["small", "medium", "large"], [4, 8, 16, 32], [1, 5, 20, 50]))
        self.assertEqual((a.scaling_max_cell_min, a.scaling_max_total_min), (5.0, 30.0))
        self.assertEqual((a.max_gb, a.min_avail_gb, a.max_sys_pct, a.max_swap_mb), (6.0, 2.0, 80.0, 512.0))
        st = cli.scaling_settings(a)
        self.assertEqual((st.max_cell_s, st.max_total_s, st.grappler), (300.0, 1800.0, "default"))
        self.assertEqual(st.max_workers, 4)
        small = cli.scaling_settings(args_for("--tests", "scaling", "--max-gb", "3"))
        self.assertEqual(small.max_workers, 2)                                   # worker cap follows the memory allowance

    def test_quick_mode_is_small_and_explicit_flags_win(self):
        a = args_for("--tests", "scaling", "--quick")
        st = cli.scaling_settings(a)
        self.assertEqual((st.pilot_pop, st.pilot_steps, st.generations, st.max_total_s), (4, (1, 20), 2, 600.0))
        self.assertEqual(args_for("--tests", "scaling", "--quick", "--scaling-pilot-pop", "6").scaling_pilot_pop, 6)

    def test_validation(self):
        bad = lambda *x: cli.validate_args(args_for("--tests", "scaling", *x))
        self.assertEqual(bad(), [])
        self.assertTrue(any("gpu-multiprocess" in e for e in bad("--scaling-allow-gpu-multiprocess")))
        self.assertTrue(any("repetitions" in e for e in bad("--scaling-confirm-reps", "1")))
        self.assertTrue(any(">= 2" in e for e in bad("--scaling-generations", "1")))
        self.assertTrue(any("max-generations" in e for e in bad("--scaling-generations", "50")))
        self.assertTrue(any("cannot exceed" in e for e in bad("--scaling-max-cell-min", "40")))

    def test_scaling_is_a_single_adaptive_job(self):
        plan = cli.plan_jobs(args_for("--tests", "scaling"))
        self.assertEqual([(t, [j.name for j in js]) for t, js in plan], [("scaling", ["scaling"])])
        self.assertEqual(cli.Job.parse("scalingblock@gpu:batched"), cli.Job("scalingblock", "gpu", "batched"))
        self.assertEqual(cli.Job.parse("scalingblock@cpu:mp4").variant, "mp4")

    def test_dry_run_reports_counts_runtime_and_exposure_without_running(self):
        a = args_for("--tests", "scaling", "--dry-run")
        buf = io.StringIO()
        from unittest import mock
        with redirect_stdout(buf), mock.patch.object(cli.subprocess, "Popen") as popen:
            rc = cli.dry_run(a, cli.limits_from(a))
        popen.assert_not_called()
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        for needle in ("NOT the full grid", "pilot 24 cells", "estimated runtime", "resource exposure", "5 min per configuration", "30 min total"):
            self.assertIn(needle, out)
        summary = D.design_summary(cli.scaling_settings(a))
        self.assertLess(summary["total_cell_runs_max"], summary["full_cartesian_points"] * 2 * 3 * 2)

    def test_preflight_refuses_when_no_paired_comparison_can_fit(self):
        from tfbench.limits import MemSample
        a = args_for("--tests", "scaling", "--quick", "--grappler", "none")
        lim = cli.limits_from(a)
        tight = MemSample(t=0.0, avail_gb=2.6, sys_pct=77.0, swap_used_gb=0.0)            # headroom ~0.6 GB, as on the failing run
        roomy = MemSample(t=0.0, avail_gb=9.0, sys_pct=40.0, swap_used_gb=0.0)
        msgs = cli.scaling_preflight(a, lim, tight)
        self.assertEqual(len(msgs), 1)
        self.assertIn("No paired CPU/GPU comparison", msgs[0])
        self.assertIn("never raised automatically", msgs[0])
        self.assertEqual(cli.scaling_preflight(a, lim, roomy), [])

    def test_dry_run_rejects_gpu_multiprocess(self):
        a = args_for("--tests", "scaling", "--dry-run", "--scaling-allow-gpu-multiprocess")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.dry_run(a, cli.limits_from(a)), 2)


class InterruptTests(unittest.TestCase):
    def test_interrupt_in_the_scaling_controller_stops_the_run_and_keeps_partial_results(self):
        from unittest import mock
        a = args_for("--tests", "scaling", "alloc", "--out-dir", "/nonexistent")
        partial = {"job": "scaling", "test": "scaling", "status": "interrupted", "_reraise": KeyboardInterrupt(), "measurements": [],
                   "data": {}, "notes": []}
        with mock.patch.object(cli, "_run_scaling", return_value=partial), mock.patch.object(cli, "run_job") as rj:
            results, interrupted = cli.supervise(a, cli.limits_from(a), "/nonexistent")
        self.assertTrue(interrupted)
        rj.assert_not_called()                                   # the next requested test (alloc) never started
        self.assertEqual([r["job"] for r in results], ["scaling"])
        self.assertEqual(results[0]["status"], "interrupted")

    def test_interrupted_controller_returns_a_partial_job(self):
        st = W.Settings()
        clock = [0.0]
        calls = []

        def runner(block, remaining):
            calls.append(block.block_id)
            if len(calls) == 3:
                raise KeyboardInterrupt()
            return SY.make_runner(st, clock)(block, remaining)
        job = R.run_scaling(st, 6.0, runner, clock=lambda: clock[0], log=lambda *_: None)
        self.assertEqual(job["status"], "interrupted")
        self.assertIsInstance(job.get("_interrupted"), KeyboardInterrupt)
        self.assertTrue(job["data"]["controller"]["cells"])                   # what finished is kept


class AnalysisTests(unittest.TestCase):
    def test_confirmed_crossover_and_model_break_even_recover_the_known_truth(self):
        st, job = synth_job()
        an = job["data"]["analysis"]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(an["crossover"]["status"], "confirmed")
        be = an["cost_model"]["break_even"]["large"]["break_even_steps"]
        self.assertAlmostEqual(be, SY.true_break_even("large"), delta=0.1 * SY.true_break_even("large"))
        self.assertIsNone(an["cost_model"]["break_even"]["small"]["break_even_steps"])      # GPU never wins on small nets
        self.assertTrue(any(p["verdict"] == "gpu_faster" for p in an["seq_points"]))
        self.assertTrue(any(p["verdict"] == "cpu_faster" for p in an["seq_points"]))

    def test_exploration_alone_never_confirms_a_crossover(self):
        st, job = synth_job()
        cells = [c for c in job["data"]["controller"]["cells"] if c["phase"] != "confirmation"]
        an = A.analyze(cells, st)
        self.assertEqual(an["crossover"]["confirmed_crossings"], [])
        self.assertIn(an["crossover"]["status"], ("candidate_unconfirmed", "none_in_tested_range"))
        self.assertTrue(all(p["verdict"] in ("unconfirmed", "no_data") for p in an["seq_points"]))

    def test_component_shares_sum_to_one(self):
        _, job = synth_job()
        for c in job["data"]["analysis"]["components"]:
            self.assertAlmostEqual(sum(c["share"].values()), 1.0, delta=0.02)

    def test_aborted_and_skipped_never_enter_statistics_and_are_listed(self):
        aborts = lambda c: c.point.tier == "large" and c.point.steps == 50 and c.strategy.mode == "seq"
        st, job = synth_job(aborts=aborts)
        an = job["data"]["analysis"]
        keys = {p["key"] for p in an["seq_points"] if p["ratio"] is not None}
        self.assertFalse(any(k.startswith("large/") and "/S50/" in k for k in keys))
        self.assertTrue(an["incomplete"]["not_completed"])
        self.assertTrue(all(x["status"] != "completed" for x in an["incomplete"]["not_completed"]))
        text = SR.markdown(job)
        self.assertIn("11.9 Incomplete, aborted, skipped", text)
        self.assertIn("never counted as a performance measurement", text)

    def test_no_gpu_gives_partial_status_and_honest_answers(self):
        st, job = synth_job(unsupported_devices=("gpu",))
        self.assertEqual(job["status"], "partial")
        self.assertIn("no paired CPU/GPU", job["reason"])
        answers = {a["q"]: a for a in job["data"]["analysis"]["answers"]}
        q1 = next(a for q, a in answers.items() if q.startswith("Is GPU"))
        self.assertIn("NOT MEASURED", q1["a"])
        q2 = next(a for q, a in answers.items() if q.startswith("At what"))
        self.assertIn("NOT MEASURED", q2["a"])
        self.assertNotIn("never reached", q2["a"])

    def test_batching_and_multiprocessing_strategies_are_analysed(self):
        _, job = synth_job()
        rows = job["data"]["analysis"]["strategies"]
        self.assertTrue(any(r["strategy"].endswith("batched") and r["population_mode"] == "homo" for r in rows))
        self.assertTrue(any(r["strategy"].startswith("cpu-mp") for r in rows))
        mp = next(r for r in rows if r["strategy"].startswith("cpu-mp"))
        self.assertAlmostEqual(mp["speedup_vs_sequential"], int(mp["strategy"][-1]) * 0.75, delta=0.4)

    def test_eight_questions_are_answered(self):
        _, job = synth_job()
        qs = [a["q"] for a in job["data"]["analysis"]["answers"]]
        self.assertEqual(len(qs), 8)
        for needle in ("faster for any", "competitive", "dominate", "batching", "memory", "evidence", "TensorFlow Metal", "uncertain"):
            self.assertTrue(any(needle in q for q in qs), needle)


class ReportTests(unittest.TestCase):
    def test_artifacts_and_main_report_integration(self):
        st, job = synth_job()
        job["data"]["synthetic"] = True
        meta = {"run_id": "t", "started_utc": "s", "finished_utc": "f", "command": "c", "hardware": {}, "software": {}, "limits": {}, "config": {}}
        rep = main_report.build_report(meta, [job])
        with tempfile.TemporaryDirectory() as d:
            paths = main_report.write_reports(rep, d)
            sp = SR.write_artifacts(d, job)
            with open(paths["json"]) as fh:
                loaded = json.load(fh, parse_constant=lambda c: self.fail(f"non-standard JSON {c}"))
            self.assertEqual(loaded["jobs"][0]["test"], "scaling")
            with open(paths["markdown"]) as fh:
                md = fh.read()
            with open(paths["csv"]) as fh:
                header = fh.readline().strip().split(",")
            with open(sp["cells_csv"]) as fh:
                cells_header = fh.readline().strip().split(",")
            for name in ("ratio_vs_steps.svg", "ratio_vs_model_size.svg", "heatmap_ratio.svg", "time_components.svg"):
                ET.parse(sp[name])
        for n in range(1, 11):
            self.assertIn(f"### 11.{n} ", md)
        self.assertIn("SYNTHETIC ILLUSTRATION", md)
        self.assertEqual(header[-4:], ["phase", "tier", "population", "steps"])
        self.assertIn("construct_s", cells_header)
        self.assertIn("unattributed_s", cells_header)

    def test_every_measurement_has_a_kind_and_json_is_strict(self):
        _, job = synth_job()
        self.assertTrue(job["measurements"])
        self.assertTrue(all(m["kind"] in common.KINDS for m in job["measurements"]))
        json.dumps(common.sanitize(job["data"]["analysis"]), allow_nan=False)

    def test_compare_tool_handles_scaling_jobs(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cmp", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "compare_reports.py"))
        cmp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmp)
        _, job = synth_job()
        m = cmp.scaling_metrics(job)
        self.assertIn("scaling_best_gpu_cpu_ratio", m)
        self.assertTrue(any(k.startswith("scaling_ratio ") for k in m))


class SchedulerTests(unittest.TestCase):
    def model(self):
        st, job = synth_job()
        an = job["data"]["analysis"]
        return st, an, job

    def test_recommendation_only_and_switches_with_steps(self):
        st, an, _ = self.model()
        m = dict(an["scheduler_model"])
        m["gains"] = {}                                                      # sequential strategies only
        lo = SC.recommend(m, params=4.2e6, layers=5, steps=5, batch=128, population=16, validation={"reliable": True})
        hi = SC.recommend(m, params=4.2e6, layers=5, steps=50, batch=128, population=16, validation={"reliable": True})
        self.assertEqual(lo["mode"], "recommendation-only")
        self.assertEqual(lo["recommendation"], "cpu-seq")
        self.assertEqual(hi["recommendation"], "gpu-seq")
        self.assertIn("never switches", lo["note"])

    def test_extrapolation_and_unvalidated_advisor_cap_confidence(self):
        st, an, _ = self.model()
        m = dict(an["scheduler_model"])
        m["gains"] = {}
        far = SC.recommend(m, params=5e9, layers=40, steps=5000, batch=128, population=16, validation={"reliable": True})
        self.assertTrue(far["extrapolating"])
        self.assertEqual(far["confidence"], "low")
        self.assertTrue(any("extrapolation" in w for w in far["warnings"]))
        near = SC.recommend(m, params=4.2e6, layers=5, steps=50, batch=128, population=16, validation={"reliable": False})
        self.assertEqual(near["confidence"], "low")
        self.assertTrue(any("capped" in w for w in near["warnings"]))

    def test_machine_mismatch_is_flagged(self):
        st, an, _ = self.model()
        m = dict(an["scheduler_model"])
        m["calibrated_on"] = {"chip": "Apple M1 Pro"}
        out = SC.recommend(m, params=4.2e6, layers=5, steps=50, batch=128, population=16, validation={"reliable": True}, machine={"chip": "Other chip"})
        self.assertTrue(any("calibrated on" in w for w in out["warnings"]))
        self.assertEqual(out["confidence"], "low")

    def test_memory_guard_disables_batched_when_it_will_not_fit(self):
        st, an, _ = self.model()
        out = SC.recommend(an["scheduler_model"], params=5e8, layers=5, steps=20, batch=128, population=32, homogeneous=True,
                           available_gb=4.0, validation={"reliable": True})
        self.assertFalse(out["memory_ok"])
        self.assertFalse(any(v != float("inf") for k, v in out["predicted_seconds"].items() if "batched" in k))

    def test_validation_is_leave_one_tier_out(self):
        _, an, _ = self.model()
        v = an["scheduler_validation"]
        self.assertIn("leave-one-tier-out", v["method"])
        self.assertTrue(set(v["per_device"]) <= {"cpu", "gpu"})
        for d in v["per_device"].values():
            self.assertTrue(set(d["per_held_out_tier"]) <= set(W.TIER_ORDER))

    def test_recommend_tool_runs_on_a_report(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("rd", os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools", "recommend_device.py"))
        rd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rd)
        _, _, job = self.model()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "report.json")
            with open(p, "w") as fh:
                json.dump(common.sanitize({"jobs": [job], "hardware": {"chip": "X"}}), fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = rd.main([p, "--params", "4.2e6", "--layers", "5", "--steps", "50", "--population", "16"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["mode"], "recommendation-only")


if __name__ == "__main__":
    unittest.main()
