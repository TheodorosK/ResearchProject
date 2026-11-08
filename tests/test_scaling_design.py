import unittest

from tfbench.scaling import controller as C, design as D, synthetic as SY, workload as W


def run_synth(st=None, **kw):
    st = st or W.Settings()
    clock = [0.0]
    calls = []
    ctrl = C.Controller(st, max_memory_gb=6.0, runner=SY.make_runner(st, clock, calls=calls, **kw), clock=lambda: clock[0], log=lambda *_: None)
    return ctrl, ctrl.run(), calls, clock


class WorkloadTests(unittest.TestCase):
    st = W.Settings()

    def test_tier_definitions_match_the_specification(self):
        for name, layers, units in (("small", (1, 2), (32, 64)), ("medium", (2, 4), (128, 256)), ("large", (4, 6), (512, 1024))):
            t = W.TIERS[name]
            self.assertEqual((t.min_layers, t.max_layers), layers)
            self.assertEqual((min(t.units), max(t.units)), units)

    def test_populations_respect_tier_ranges_and_are_deterministic(self):
        for tier in W.TIER_ORDER:
            pt = W.Point(tier, 16, 5)
            pop = W.initial_population(pt, self.st)
            self.assertEqual(len(pop), 16)
            t = W.TIERS[tier]
            for g in pop:
                self.assertTrue(t.min_layers <= len(g.layers) <= t.max_layers)
                self.assertTrue(all(l.units in t.units for l in g.layers))
            again = W.initial_population(pt, self.st)
            self.assertEqual([x.describe() for x in pop], [x.describe() for x in again])

    def test_paired_workloads_are_identical_across_devices(self):
        """The signature (genomes, data spec, seed, budget) has NO dependence on the strategy; cells pair by Point."""
        pt = W.Point("medium", 8, 20)
        cells = [W.Cell(pt, W.Strategy(d, m, w)) for d, m, w in (("cpu", "seq", 1), ("gpu", "seq", 1), ("cpu", "mp", 2), ("gpu", "batched", 1))]
        sigs = {W.workload_signature(c.point, self.st) for c in cells}
        self.assertEqual(len(sigs), 1)
        self.assertNotEqual(W.workload_signature(pt, self.st), W.workload_signature(W.Point("medium", 8, 21), self.st))
        self.assertNotEqual(W.workload_signature(pt, self.st), W.workload_signature(W.Point("medium", 8, 20, "homo"), self.st))
        other_seed = W.Settings(seed=1)
        self.assertNotEqual(W.workload_signature(pt, self.st), W.workload_signature(pt, other_seed))

    def test_homogeneous_populations_are_batch_compatible_and_offspring_stay_so(self):
        import random
        from tfbench import genetics as G
        pt = W.Point("medium", 8, 5, "homo")
        pop = W.initial_population(pt, self.st)
        self.assertEqual(len(W.bucket_genomes(pop)), 1)
        self.assertEqual(W.padding_overhead(pop, self.st.in_dim), 1.0)
        space = W.make_space(pt, self.st)
        rng = random.Random(0)
        off, _ = G.next_generation(pop, [0.1 * i for i in range(8)], rng, space, elite_k=2, tournament_k=3, crossover_rate=0.7, next_uid=8, generation=1)
        self.assertEqual(len(W.bucket_genomes([o.genome for o in off])), 1)

    def test_buckets_and_padding(self):
        from tfbench import genetics as G
        a = G.Genome([G.LayerGene(32, "relu"), G.LayerGene(64, "tanh")], 1e-3)
        b = G.Genome([G.LayerGene(64, "relu"), G.LayerGene(32, "tanh")], 1e-3)
        c = G.Genome([G.LayerGene(32, "tanh")], 1e-3)
        self.assertEqual(sorted(map(sorted, W.bucket_genomes([a, b, c]))), [[0, 1], [2]])
        self.assertGreater(W.padding_overhead([a, b], 32), 1.0)

    def test_evaluation_counts_and_batch_indices(self):
        self.assertEqual(W.elite_count(4), 1)
        self.assertEqual(W.n_evaluations(8, 3), 8 + 2 * 6)
        idx = W.batch_indices(3, 1000, 64, 40)
        self.assertEqual(idx.shape, (40, 64))
        self.assertTrue((idx >= 0).all() and (idx < 1000).all())
        self.assertTrue((idx == W.batch_indices(3, 1000, 64, 40)).all())
        self.assertEqual(len(set(idx[:15].ravel())), 15 * 64)                     # one pass = a permutation: no repeats


class DesignTests(unittest.TestCase):
    st = W.Settings()

    def test_full_cartesian_is_never_the_plan(self):
        s = D.design_summary(self.st)
        self.assertEqual(s["full_cartesian_points"], 3 * 4 * 4)
        self.assertEqual(s["pilot_cells"], 3 * 4 * 2)
        self.assertLess(s["total_cell_runs_max"], 3 * 4 * 4 * 2 * 2)              # far below the exhaustive sweep x strategies
        self.assertLess(D.design_summary(W.Settings(quick=True, pilot_pop=4, pilot_steps=(1, 20), generations=2, confirm_points=1))["pilot_cells"], 24)

    def test_pilot_pairs_cpu_and_gpu_on_identical_points(self):
        cells = D.pilot_cells(self.st)
        by = {}
        for c in cells:
            by.setdefault(c.point.key, set()).add(c.strategy.device)
        self.assertTrue(all(v == {"cpu", "gpu"} for v in by.values()))
        self.assertTrue(all(c.strategy.mode == "seq" and c.point.mode == "hetero" for c in cells))

    def test_candidates_include_both_sides_of_a_sign_change_and_the_best(self):
        def e(tier, S, r):
            return {"point": {"tier": tier, "population": 8, "steps": S, "mode": "hetero", "generations": 3}, "key": f"{tier}/{S}", "ratio": r}
        table = [e("large", 1, 0.2), e("large", 5, 0.4), e("large", 20, 0.9), e("large", 50, 1.6), e("small", 50, 0.1)]
        keys = [c["key"] for c in D.choose_candidates(table, 4)]
        self.assertEqual(keys[0], "large/50")
        self.assertIn("large/20", keys)                                              # other side of the crossing
        self.assertEqual(D.choose_candidates([], 2), [])
        hopeless = [e("small", 1, 0.05), e("large", 50, 0.3)]
        self.assertEqual(D.choose_candidates(hopeless, 1)[0]["key"], "large/50")     # best point is confirmed even if hopeless

    def test_skip_reasons_time_memory_and_pruning(self):
        cost, st = D.CostModel(), self.st
        big = W.Cell(W.Point("large", 32, 50, "hetero", 3), W.Strategy("gpu"))
        self.assertIn("exceeds the per-configuration limit", D.skip_reason(big, W.Settings(max_cell_s=30), cost, 6.0, {}))
        mem = W.Cell(W.Point("large", 32, 5, "homo", 3), W.Strategy("gpu", "batched"))
        self.assertIn("estimated memory", D.skip_reason(mem, st, cost, 2.0, {}))
        ok = W.Cell(W.Point("small", 4, 5), W.Strategy("cpu"))
        self.assertIsNone(D.skip_reason(ok, st, cost, 6.0, {}))
        blocked = {("cpu-seq", "small"): 0.0}
        self.assertIn("already aborted", D.skip_reason(ok, st, cost, 6.0, blocked))

    def test_expansion_respects_budget_and_skips_gpu_multiprocessing(self):
        cands = [{"point": {"tier": "medium", "population": 8, "steps": 20, "mode": "hetero", "generations": 3}, "key": "k", "ratio": 1.0}]
        cells, skipped = D.plan_expansion(cands, self.st, [], D.CostModel(), 5.0, 6.0)
        self.assertTrue(sum(D.CostModel().predict(c.point, c.strategy, self.st) for c in cells) <= 5.0 + 1e-9)
        self.assertTrue(any("budget" in s["reason"] for s in skipped))
        cells, _ = D.plan_expansion(cands, self.st, [], D.CostModel(), 1e6, 6.0)
        self.assertFalse(any(c.strategy.mode == "mp" and c.strategy.device == "gpu" for c in cells))
        self.assertTrue(all(c.strategy.workers <= self.st.max_workers for c in cells))
        self.assertTrue(any(c.strategy.mode == "mp" for c in cells))
        no_mp, _ = D.plan_expansion(cands, W.Settings(skip_mp=True, skip_batched=True), [], D.CostModel(), 1e6, 6.0)
        self.assertTrue(all(c.strategy.mode == "seq" for c in no_mp))


class ControllerTests(unittest.TestCase):
    def test_adaptive_phases_pilot_expansion_confirmation(self):
        ctrl, res, calls, clock = run_synth()
        phases = {r["phase"] for r in res["cells"]}
        self.assertEqual(phases, {"pilot", "expansion", "confirmation"})
        n_conf = [r for r in res["cells"] if r["phase"] == "confirmation"]
        self.assertGreaterEqual(max(r["rep"] for r in n_conf), 1)                      # repeated, independent repetitions
        self.assertLess(len(res["cells"]), 3 * 4 * 4 * 2 * 3)                          # not the Cartesian product
        self.assertLessEqual(clock[0], W.Settings().max_total_s * 1.05)
        self.assertEqual(res["status"], "completed")
        self.assertTrue(res["phase_log"]["candidates"])

    def test_confirmed_batched_strategy_is_paired_across_devices(self):
        _, res, _, _ = run_synth(W.Settings(skip_mp=True, pilot_pop=4, pops=(4, 8)))   # batching is the best gain; P=8 is an adjacent size
        plan = res["phase_log"]["confirmation_plan"]
        batched_entries = [names for _, names in plan if any(n.endswith("batched") for n in names)]
        self.assertTrue(batched_entries)
        for names in batched_entries:
            self.assertIn("cpu-batched", names)                   # identical-strategy pair, so the device ratio gets a CI
            self.assertIn("gpu-batched", names)
        conf_batched = {r["strategy"]["device"] for r in res["cells"] if r["phase"] == "confirmation" and r["strategy"]["mode"] == "batched"}
        self.assertEqual(conf_batched, {"cpu", "gpu"})
        pops = {r["point"]["population"] for r in res["cells"] if r["phase"] == "confirmation" and r["strategy"]["mode"] == "batched"}
        self.assertGreaterEqual(len(pops), 2)                      # adjacent population sizes are confirmed, so a P-axis crossing can be

    def test_mp_is_planned_at_the_homogeneous_point_too(self):
        cands = [{"point": {"tier": "large", "population": 8, "steps": 20, "mode": "hetero", "generations": 3}, "key": "k", "ratio": 0.4}]
        cells, _ = D.plan_expansion(cands, W.Settings(), [], D.CostModel(), 1e6, 6.0)
        mp_modes = {c.point.mode for c in cells if c.strategy.mode == "mp"}
        self.assertEqual(mp_modes, {"hetero", "homo"})

    def test_confirmation_uses_fresh_blocks_per_repetition_and_alternates_device_order(self):
        _, res, _, _ = run_synth()
        conf = [b for b in res["blocks"] if b["phase"] == "confirmation"]
        self.assertGreaterEqual(len({b["rep"] for b in conf}), 2)
        first_by_rep = {}
        for b in conf:
            first_by_rep.setdefault(b["rep"], b["device"])
        self.assertGreater(len(set(first_by_rep.values())), 1)

    def test_total_budget_is_enforced_and_skips_are_recorded(self):
        st = W.Settings(max_total_s=240.0)
        _, res, _, clock = run_synth(st)
        self.assertLessEqual(clock[0], 240.0 * 1.6)
        self.assertTrue(res["skipped"])
        self.assertTrue(any("budget" in s["reason"] for s in res["skipped"]))

    def test_aborted_cells_are_recorded_pruned_and_never_counted_as_measurements(self):
        aborts = lambda c: c.point.tier == "large" and c.point.steps >= 20 and c.strategy.mode == "seq"
        ctrl, res, _, _ = run_synth(aborts=aborts)
        aborted = [r for r in res["cells"] if r["status"] == "aborted"]
        self.assertTrue(aborted)
        self.assertTrue(all(r["ips"] is None for r in aborted))
        table = D.seq_ratio_table(res["cells"])
        keys = {e["key"] for e in table}
        self.assertFalse(any(k.startswith("large/") and "/S20/" in k or "/S50/" in k and k.startswith("large/") for k in keys))
        self.assertTrue(res["blocked"])

    def test_unavailable_gpu_is_recorded_once_without_repeated_spawns(self):
        _, res, calls, _ = run_synth(unsupported_devices=("gpu",))
        gpu_calls = [c for c in calls if "-gpu-" in c]
        self.assertEqual(len(gpu_calls), 1)
        self.assertIn("gpu-*", res["unusable_strategies"])           # device-wide, not just gpu-seq
        self.assertTrue(any(r["status"] == "unsupported" for r in res["cells"]))
        self.assertEqual(res["status"], "completed")
        self.assertTrue(any(r["strategy"]["mode"] == "mp" for r in res["cells"]))      # CPU-only exploration still happens

    def test_soft_placement_gpu_results_are_not_usable_for_ratios(self):
        _, res, _, _ = run_synth(soft_gpu=True)
        self.assertEqual(D.seq_ratio_table(res["cells"]), [])

    def test_quick_mode_is_small(self):
        st = W.Settings(quick=True, pilot_pop=4, pilot_steps=(1, 20), generations=2, confirm_points=1, pops=(4, 8), workers=(2,))
        ctrl, res, calls, clock = run_synth(st)
        self.assertLess(len(res["cells"]), D.design_summary(st)["total_cell_runs_max"] + 1)        # never more than the announced upper bound
        self.assertLess(len(res["cells"]), 60)
        self.assertLess(clock[0], 11 * 60)


if __name__ == "__main__":
    unittest.main()
