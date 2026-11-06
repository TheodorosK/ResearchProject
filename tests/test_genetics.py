import math
import random
import unittest

import numpy as np

from tfbench import genetics as G

SPACE = G.SearchSpace()


def rand_pop(n, seed=0, space=SPACE):
    rng = random.Random(seed)
    return [G.random_genome(rng, space, uid=i) for i in range(n)]


class ValidityTests(unittest.TestCase):
    def test_random_genomes_valid(self):
        for g in rand_pop(300):
            self.assertEqual(G.validate_genome(g, SPACE), [])

    def test_repair_fixes_everything(self):
        bad = G.Genome(layers=[G.LayerGene(7, "nope")] * 9, learning_rate=float("nan"),
                       weight_sigma=-1, weight_rate=5, arch_mut_rate=float("inf"))
        self.assertTrue(G.validate_genome(bad, SPACE))
        G.repair(bad, SPACE)
        self.assertEqual(G.validate_genome(bad, SPACE), [])
        self.assertEqual(len(bad.layers), SPACE.max_layers)

    def test_repair_pads_empty(self):
        g = G.Genome(layers=[], learning_rate=1e-3)
        G.repair(g, SPACE)
        self.assertEqual(G.validate_genome(g, SPACE), [])

    def test_space_problems(self):
        self.assertTrue(G.SearchSpace(min_layers=3, max_layers=2).problems())
        self.assertTrue(G.SearchSpace(unit_choices=()).problems())
        self.assertEqual(SPACE.problems(), [])


class OperatorTests(unittest.TestCase):
    def test_fuzz_mutation_and_crossover_always_valid(self):
        rng = random.Random(1)
        pop = rand_pop(20, seed=2)
        uid = 100
        for _ in range(2000):
            a, b = rng.sample(pop, 2)
            child, info = G.crossover(a, b, rng, SPACE, uid, 1)
            self.assertEqual(G.validate_genome(child, SPACE), [], info)
            child, _ = G.mutate_architecture(child, rng, SPACE)
            child = G.mutate_numeric(child, rng, SPACE)
            self.assertEqual(G.validate_genome(child, SPACE), [])
            m, _ = G.mutate(a, rng, SPACE, uid, 1)
            self.assertEqual(G.validate_genome(m, SPACE), [])
            uid += 1
            pop[rng.randrange(len(pop))] = child

    def test_mutation_never_exceeds_depth_bounds(self):
        rng = random.Random(3)
        full = G.Genome([G.LayerGene(16, "relu")] * SPACE.max_layers, 1e-3, arch_mut_rate=0.9)
        minimal = G.Genome([G.LayerGene(16, "relu")] * SPACE.min_layers, 1e-3, arch_mut_rate=0.9)
        for _ in range(500):
            g, op = G.mutate_architecture(full.copy(), rng, SPACE)
            self.assertLessEqual(len(g.layers), SPACE.max_layers)
            self.assertNotEqual(op, "add_layer")
            g, op = G.mutate_architecture(minimal.copy(), rng, SPACE)
            self.assertGreaterEqual(len(g.layers), SPACE.min_layers)
            self.assertNotEqual(op, "remove_layer")

    def test_arch_rate_zero_like_means_rare_change(self):
        rng = random.Random(4)
        lo = SPACE.arch_rate_range[0]
        g = G.Genome([G.LayerGene(32, "relu")] * 2, 1e-3, arch_mut_rate=lo)
        changed = sum(G.mutate_architecture(g.copy(), rng, SPACE)[1] is not None for _ in range(2000))
        self.assertLess(changed / 2000, lo * 2)

    def test_mutation_is_deterministic_for_seed(self):
        g = rand_pop(1)[0]
        r1 = G.mutate(g, random.Random(9), SPACE, 5, 1)[0]
        r2 = G.mutate(g, random.Random(9), SPACE, 5, 1)[0]
        self.assertEqual(r1, r2)

    def test_mutation_does_not_modify_parent(self):
        g = rand_pop(1)[0]
        before = (list(g.layers), g.learning_rate)
        for s in range(50):
            G.mutate(g, random.Random(s), SPACE, 5, 1)
        self.assertEqual((list(g.layers), g.learning_rate), before)

    def test_crossover_same_depth_takes_genes_from_parents(self):
        a = G.Genome([G.LayerGene(8, "relu"), G.LayerGene(16, "relu")], 1e-3)
        b = G.Genome([G.LayerGene(64, "tanh"), G.LayerGene(128, "tanh")], 1e-2, uid=1)
        rng = random.Random(0)
        for _ in range(100):
            child, info = G.crossover(a, b, rng, SPACE, 9, 1)
            self.assertEqual(info["mode"], "same_depth_uniform")
            self.assertTrue(G.compatible(a, b))
            for i, l in enumerate(child.layers):
                self.assertIn(l, (a.layers[i], b.layers[i]))
            self.assertTrue(min(a.learning_rate, b.learning_rate) * 0.999 <= child.learning_rate
                            <= max(a.learning_rate, b.learning_rate) * 1.001)

    def test_crossover_incompatible_depth_is_aligned_and_valid(self):
        a = G.Genome([G.LayerGene(8, "relu")], 1e-3, uid=0)
        b = G.Genome([G.LayerGene(u, "tanh") for u in (16, 32, 64, 128)], 1e-2, uid=1)
        self.assertFalse(G.compatible(a, b))
        rng = random.Random(0)
        depths = set()
        for _ in range(200):
            child, info = G.crossover(a, b, rng, SPACE, 9, 1)
            self.assertEqual(info["mode"], "depth_aligned")
            depths.add(len(child.layers))
            self.assertEqual(G.validate_genome(child, SPACE), [])
        self.assertEqual(depths, {1, 4})        # child depth equals one parent's depth

    def test_aligned_index(self):
        self.assertEqual(G._aligned_index(0, 1, 4), 0)
        self.assertEqual(G._aligned_index(0, 4, 2), 0)
        self.assertEqual(G._aligned_index(3, 4, 2), 1)


class SelectionTests(unittest.TestCase):
    def test_tournament_full_size_picks_best(self):
        scores = [0.1, 0.9, 0.5, None, float("nan")]
        for s in range(20):
            self.assertEqual(G.tournament_select(random.Random(s), scores, k=5), 1)

    def test_tournament_never_picks_failure_over_success(self):
        scores = [None, 0.2, None, None]
        for s in range(50):
            self.assertEqual(G.tournament_select(random.Random(s), scores, k=4), 1)

    def test_tournament_all_failed_still_returns_index(self):
        i = G.tournament_select(random.Random(0), [None, None, None], k=2)
        self.assertIn(i, (0, 1, 2))

    def test_elites_top_k_and_exclude_failures(self):
        self.assertEqual(G.elite_indices([0.3, None, 0.9, 0.5], 2), [2, 3])
        self.assertEqual(G.elite_indices([None, None], 2), [])
        self.assertEqual(G.elite_indices([0.5, 0.5], 1), [0])       # stable tie-break

    def test_next_generation_contract(self):
        pop = rand_pop(10, seed=5)
        scores = [random.Random(i).random() for i in range(10)]
        rng = random.Random(0)
        off, uid = G.next_generation(pop, scores, rng, SPACE, elite_k=2, tournament_k=3,
                                     crossover_rate=0.7, next_uid=10, generation=1)
        self.assertEqual(len(off), 10)
        elites = [o for o in off if o.is_elite]
        self.assertEqual(len(elites), 2)
        best = sorted(range(10), key=lambda i: -scores[i])[:2]
        self.assertEqual({o.genome.uid for o in elites}, {pop[i].uid for i in best})
        uids = [o.genome.uid for o in off]
        self.assertEqual(len(set(uids)), 10)
        self.assertEqual(uid, 10 + 8)
        for o in off:
            self.assertEqual(G.validate_genome(o.genome, SPACE), [])

    def test_next_generation_elite_larger_than_population(self):
        pop = rand_pop(3)
        off, _ = G.next_generation(pop, [0.1, 0.2, 0.3], random.Random(0), SPACE, elite_k=99,
                                   tournament_k=2, crossover_rate=1.0, next_uid=3, generation=1)
        self.assertEqual(len(off), 3)
        self.assertEqual(sum(o.is_elite for o in off), 2)


class WeightTests(unittest.TestCase):
    @staticmethod
    def make(dims, fill):
        out = []
        for i, o in dims:
            out += [np.full((i, o), fill, np.float32), np.full((o,), fill, np.float32)]
        return out

    def test_count_params(self):
        g = G.Genome([G.LayerGene(8, "relu"), G.LayerGene(4, "relu")], 1e-3)
        self.assertEqual(G.layer_dims(g, 5, 1), [(5, 8), (8, 4), (4, 1)])
        self.assertEqual(G.count_params(g, 5, 1), 5 * 8 + 8 + 8 * 4 + 4 + 4 + 1)

    def test_inherit_identical_architecture_copies_everything(self):
        dims = [(5, 8), (8, 1)]
        parent, child = self.make(dims, 2.0), self.make(dims, 0.0)
        w, frac = G.inherit_weights(parent, 1, child, 1)
        self.assertAlmostEqual(frac, 1.0)
        for a, b in zip(w, parent):
            np.testing.assert_array_equal(a, b)

    def test_inherit_overlap_when_wider_or_narrower(self):
        parent = self.make([(5, 8), (8, 1)], 2.0)
        wide = self.make([(5, 16), (16, 1)], 0.0)
        w, frac = G.inherit_weights(parent, 1, wide, 1)
        self.assertTrue((w[0][:, :8] == 2.0).all() and (w[0][:, 8:] == 0.0).all())
        self.assertTrue((w[2][:8] == 2.0).all() and (w[2][8:] == 0.0).all())
        self.assertGreater(frac, 0)
        narrow = self.make([(5, 4), (4, 1)], 0.0)
        w, _ = G.inherit_weights(parent, 1, narrow, 1)
        self.assertTrue((w[0] == 2.0).all() and w[0].shape == (5, 4))

    def test_inherit_different_depth_skips_output_layer(self):
        parent = self.make([(5, 8), (8, 1)], 2.0)
        child = self.make([(5, 8), (8, 8), (8, 1)], 0.0)
        w, _ = G.inherit_weights(parent, 1, child, 2)
        self.assertTrue((w[0] == 2.0).all())            # first hidden layer inherited
        self.assertTrue((w[2] == 0.0).all())            # new hidden layer: fresh init
        self.assertTrue((w[4] == 0.0).all())            # output layer NOT inherited (depth differs)

    def test_inherit_does_not_mutate_inputs(self):
        parent, child = self.make([(3, 4), (4, 1)], 1.0), self.make([(3, 4), (4, 1)], 0.0)
        G.inherit_weights(parent, 1, child, 1)
        self.assertTrue(all((c == 0).all() for c in child))

    def test_perturb_weights(self):
        w = [np.random.default_rng(0).normal(size=(6, 6)).astype(np.float32)]
        same = G.perturb_weights(w, np.random.default_rng(1), sigma=0.1, rate=0.0)
        np.testing.assert_array_equal(same[0], w[0])
        a = G.perturb_weights(w, np.random.default_rng(1), sigma=0.1, rate=1.0)
        b = G.perturb_weights(w, np.random.default_rng(1), sigma=0.1, rate=1.0)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[0].shape, w[0].shape)
        self.assertFalse(np.array_equal(a[0], w[0]))


if __name__ == "__main__":
    unittest.main()
