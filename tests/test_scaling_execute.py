"""TensorFlow-dependent tests of the scaling executor (skipped when TensorFlow is not installed)."""
import importlib.util
import unittest

from tfbench.limits import LimitExceeded
from tfbench.scaling import workload as W

HAS_TF = importlib.util.find_spec("tensorflow") is not None


class FakeMonitor:
    def __init__(self, aborted=False, reason=""):
        self.aborted, self.reason = aborted, reason
        from tfbench.limits import Baseline
        self.baseline = Baseline(0.0, 8.0, 0.0)

    def window_peaks(self, reset=True):
        return {"max_extra_gb": 0.1, "min_avail_gb": 6.0, "max_sys_pct": 50.0, "max_swap_growth_mb": 0.0, "max_rss_gb": 1.0}


@unittest.skipUnless(HAS_TF, "tensorflow not installed")
class ExecuteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tfbench import tfenv
        from tfbench.scaling import execute
        cls.tf = tfenv.configure("cpu", "strict", "none")
        cls.E = execute
        cls.st = W.Settings(train_n=512, val_n=256, generations=2)
        cls.data = W.dataset_for(cls.st)

    def genomes(self):
        from tfbench import genetics as G
        return [G.Genome([G.LayerGene(32, "relu"), G.LayerGene(64, "tanh")], 2e-3, uid=11),
                G.Genome([G.LayerGene(64, "relu"), G.LayerGene(32, "tanh")], 1e-3, uid=12),
                G.Genome([G.LayerGene(48, "relu"), G.LayerGene(48, "tanh")], 3e-3, uid=13)]

    def test_batched_padded_masked_training_equals_sequential(self):
        gs = self.genomes()                                   # three DIFFERENT widths -> zero-padding + masks are exercised
        for steps in (1, 5):
            seq = [self.E.eval_individual(self.tf, "cpu", g, self.data, self.st, steps)["val_r2"] for g in gs]
            bat = [r["val_r2"] for r in self.E.eval_bucket_batched(self.tf, "cpu", gs, self.data, self.st, steps)]
            for a, b in zip(seq, bat):
                self.assertAlmostEqual(a, b, places=4)
        self.assertGreater(self.E.eval_bucket_batched(self.tf, "cpu", gs, self.data, self.st, 2)[0]["padded_ratio"], 1.0)

    def test_batched_population_buckets_and_coverage(self):
        from tfbench import genetics as G
        gs = self.genomes() + [G.Genome([G.LayerGene(32, "tanh")], 1e-3, uid=14)]       # one singleton bucket
        out = self.E.eval_batched_population(self.tf, "cpu", gs, self.data, self.st, 2, lambda: None)
        self.assertEqual([r["batched"] for r in out], [True, True, True, False])
        self.assertTrue(all(r["val_r2"] is not None for r in out))

    def test_component_decomposition_sums_to_wall(self):
        cell = W.Cell(W.Point("small", 4, 5, "hetero", 2), W.Strategy("cpu"), "pilot")
        ctx = self.E.Ctx(self.st, "cpu", None, self.tf)
        r = self.E.run_cell(ctx, cell, 120)
        self.assertEqual(r["status"], "completed")
        self.assertEqual(r["evaluations"], W.n_evaluations(4, 2))
        c = r["comps"]
        self.assertAlmostEqual(sum(v for v in c.values() if v is not None), r["wall_s"], delta=0.02 * r["wall_s"] + 1e-3)
        self.assertGreaterEqual(c["unattributed"], -1e-3)
        for k in ("construct", "trace", "first_exec", "train", "fitness", "teardown"):
            self.assertGreaterEqual(c[k], 0.0)
        self.assertIn("teardown", c)
        self.assertAlmostEqual(r["ips"], r["evaluations"] / r["wall_s"])

    def test_single_step_budget_has_no_train_component(self):
        r = self.E.run_cell(self.E.Ctx(self.st, "cpu", None, self.tf), W.Cell(W.Point("small", 4, 1, "hetero", 2), W.Strategy("cpu")), 120)
        self.assertEqual(r["comps"]["train"], 0.0)
        self.assertGreater(r["comps"]["first_exec"], 0.0)
        self.assertIsNone(r["train_steps_per_s"])

    def test_cell_time_limit_is_aborted_not_measured(self):
        r = self.E.run_cell(self.E.Ctx(self.st, "cpu", None, self.tf), W.Cell(W.Point("medium", 8, 20, "hetero", 3), W.Strategy("cpu")), 0.05)
        self.assertEqual(r["status"], "aborted")
        self.assertIsNone(r["ips"])
        self.assertIn("time limit", r["reason"])

    def test_resource_limit_stops_the_cell_and_is_flagged_for_the_block(self):
        ctx = self.E.Ctx(self.st, "cpu", FakeMonitor(True, "extra system memory 7 GB >= limit 6 GB"), self.tf)
        r = self.E.run_cell(ctx, W.Cell(W.Point("small", 4, 5, "hetero", 2), W.Strategy("cpu")), 120)
        self.assertEqual(r["status"], "aborted")
        self.assertTrue(r["_limit"])
        self.assertIsNone(r["ips"])
        self.assertIn("extra system memory", r["reason"])

    def test_paired_cpu_cells_share_workload_and_fitness(self):
        pt = W.Point("small", 4, 5, "hetero", 2)
        a = self.E.run_cell(self.E.Ctx(self.st, "cpu", None, self.tf), W.Cell(pt, W.Strategy("cpu")), 120)
        b = self.E.run_cell(self.E.Ctx(self.st, "cpu", None, self.tf), W.Cell(pt, W.Strategy("cpu", "batched")), 120)
        self.assertEqual(a["signature"], b["signature"])
        self.assertAlmostEqual(a["best_val_r2"], b["best_val_r2"], places=3)

    def test_device_verification_on_cpu(self):
        from tfbench.common import Recorder
        rec = Recorder("x")
        self.assertIsNone(self.E.verify_device(self.tf, "cpu", self.st, rec))
        self.assertTrue(rec.data["device_evidence"]["verified"])
        self.assertIsNone(self.E.selfcheck_batched(self.tf, "cpu", self.st, rec))

    def test_gpu_request_without_gpu_never_falls_back_to_cpu(self):
        from tfbench import tfenv
        if self.tf.config.list_physical_devices("GPU"):
            self.skipTest("a GPU is present")
        with self.assertRaises(tfenv.DeviceUnavailable):
            tfenv.configure("gpu", "strict", "none")

    def test_multiprocessing_spawn_pool_matches_sequential(self):
        from tfbench.scaling.pool import MPEvaluator
        pt = W.Point("small", 4, 3, "hetero", 2)
        seq = self.E.run_cell(self.E.Ctx(self.st, "cpu", None, self.tf), W.Cell(pt, W.Strategy("cpu")), 120)
        ev = MPEvaluator(2, self.st, rss_cap_total_gb=6.0)
        try:
            mp = self.E.run_cell(self.E.Ctx(self.st, "cpu", None, None, evaluator=ev), W.Cell(pt, W.Strategy("cpu", "mp", 2)), 120)
        finally:
            ev.close()
        self.assertEqual(mp["status"], "completed")
        self.assertAlmostEqual(seq["best_val_r2"], mp["best_val_r2"], places=3)
        self.assertEqual(mp["mp"]["threads_per_worker"], 1)
        self.assertIsNone(mp["comps"]["unattributed"])                       # worker-seconds basis: no wall attribution
        self.assertIn("worker-seconds", mp["comps_basis"])

    def test_pool_aborts_and_terminates_workers_when_rss_cap_exceeded(self):
        from tfbench.scaling.pool import MPEvaluator
        ev = MPEvaluator(2, self.st, rss_cap_total_gb=0.05)                   # absurdly small cap
        try:
            with self.assertRaises(LimitExceeded):
                ev.evaluate(W.initial_population(W.Point("medium", 8, 20, "hetero", 1), self.st), 20, lambda: None)
        finally:
            ev.close()


class PoolCapTests(unittest.TestCase):
    def test_worker_cap_follows_the_memory_allowance(self):
        from tfbench.scaling.pool import max_workers_by_memory
        self.assertEqual(max_workers_by_memory(6.0, requested=4), 4)
        self.assertEqual(max_workers_by_memory(4.0, requested=4), 2)          # 2 GB for workers / 0.75 GB each
        self.assertEqual(max_workers_by_memory(6.0, requested=2), 2)
        self.assertEqual(max_workers_by_memory(1.0, requested=4), 1)
        self.assertEqual(max_workers_by_memory(32.0, requested=4), 4)


if __name__ == "__main__":
    unittest.main()
