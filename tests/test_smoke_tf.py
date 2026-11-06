"""TensorFlow-dependent smoke tests (skipped automatically when TensorFlow is not installed)."""
import importlib.util
import random
import unittest
from types import SimpleNamespace

import numpy as np

from tfbench import evolution, genetics as G

HAS_TF = importlib.util.find_spec("tensorflow") is not None


class DatasetTests(unittest.TestCase):          # numpy only
    def test_dataset_is_reproducible_and_split(self):
        a = evolution.make_dataset(100, 40, 30, 8, seed=7)
        b = evolution.make_dataset(100, 40, 30, 8, seed=7)
        c = evolution.make_dataset(100, 40, 30, 8, seed=8)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertFalse(np.array_equal(a["x_train"], c["x_train"]))
        self.assertEqual((len(a["x_train"]), len(a["x_val"]), len(a["x_test"])), (100, 40, 30))
        self.assertAlmostEqual(float(a["y_train"].mean()), 0.0, places=4)     # normalised by TRAIN stats

    def test_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(evolution.r2_score(y, y), 1.0)
        self.assertAlmostEqual(evolution.r2_score(y, np.full(4, y.mean())), 0.0)


@unittest.skipUnless(HAS_TF, "tensorflow not installed")
class EvaluateTests(unittest.TestCase):
    def setUp(self):
        from tfbench import tfenv
        self.tf = tfenv.import_tf()
        self.data = evolution.make_dataset(256, 64, 64, 8, seed=1)
        self.args = SimpleNamespace(seed=3, ne_epochs=2, ne_batch=64)

    def genome(self, uid=0):
        return G.Genome([G.LayerGene(16, "relu"), G.LayerGene(8, "tanh")], 5e-3, uid=uid)

    def test_evaluate_ok_and_reproducible(self):
        r1 = evolution.evaluate(self.tf, "cpu", self.genome(), self.data, self.args)
        r2 = evolution.evaluate(self.tf, "cpu", self.genome(), self.data, self.args)
        self.assertEqual(r1.status, "ok")
        self.assertTrue(np.isfinite(r1.val_r2) and np.isfinite(r1.train_r2))
        self.assertEqual(r1.params, G.count_params(self.genome(), 8, 1))
        self.assertAlmostEqual(r1.val_r2, r2.val_r2, places=4)
        self.assertEqual(len(r1.weights), 2 * (2 + 1))

    def test_simple_adam_matches_keras_legacy_adam(self):
        from tfbench import tfenv
        tf = self.tf
        rng = np.random.default_rng(0)
        x, y = tf.constant(rng.normal(size=(32, 5)).astype("float32")), tf.constant(rng.normal(size=(32, 1)).astype("float32"))
        outs = []
        for kind in ("simple", "keras"):
            w = [rng.normal(size=s_).astype("float32") * 0 + 0.1 for s_ in ((5, 4), (4,), (4, 1), (1,))]
            m = tf.keras.Sequential([tf.keras.layers.Input((5,)), tf.keras.layers.Dense(4, activation="tanh"),
                                     tf.keras.layers.Dense(1)])
            m.set_weights(w)
            opt, _ = tfenv.make_optimizer(tf, 1e-2, m.trainable_variables, kind)
            for _ in range(6):
                with tf.GradientTape() as tape:
                    loss = tf.reduce_mean(tf.square(y - m(x, training=True)))
                opt.apply_gradients(zip(tape.gradient(loss, m.trainable_variables), m.trainable_variables))
            outs.append(m.get_weights())
        for a, b in zip(*outs):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6)

    def test_loss_trace_recorded(self):
        r = evolution.evaluate(self.tf, "cpu", self.genome(), self.data, self.args, trace_steps=4)
        self.assertEqual(len(r.loss_trace), 4)
        self.assertLess(r.loss_trace[-1], r.loss_trace[0] * 1.5)

    def test_failed_evaluation_is_recorded_not_raised(self):
        g = self.genome()
        g.learning_rate = 1e30                      # diverges -> non-finite fitness
        r = evolution.evaluate(self.tf, "cpu", g, self.data, self.args)
        self.assertEqual(r.status, "failed")
        self.assertIsNone(r.val_r2)

    def test_weight_inheritance_runs_for_every_depth_combination(self):
        rng = random.Random(0)
        space = G.SearchSpace()
        parent = evolution.evaluate(self.tf, "cpu", self.genome(), self.data, self.args)
        for i in range(8):
            child = G.random_genome(rng, space, uid=10 + i)
            r = evolution.evaluate(self.tf, "cpu", child, self.data, self.args, donor=parent, donor_depth=2)
            self.assertEqual(r.status, "ok", r.error)
            self.assertGreaterEqual(r.inherited_fraction, 0.0)
            self.assertLessEqual(r.inherited_fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
