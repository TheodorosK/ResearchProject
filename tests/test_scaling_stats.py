import math
import unittest

from tfbench.scaling import stats as S


class StatsTests(unittest.TestCase):
    def test_t_crit(self):
        self.assertAlmostEqual(S.t_crit(2), 4.303)
        self.assertAlmostEqual(S.t_crit(1000), 1.96)
        self.assertTrue(4.303 > S.t_crit(2.5) > 3.182)             # interpolated
        self.assertEqual(S.t_crit(0.5), float("inf"))

    def test_median_cv(self):
        m = S.median_cv([10, 10, 10])
        self.assertEqual((m["median"], m["cv"]), (10, 0.0))
        self.assertIsNone(S.median_cv([5])["cv"])
        self.assertEqual(S.median_cv([])["n"], 0)

    def test_ci_matches_hand_computation(self):
        cpu, gpu = [10.0, 10.2, 9.8], [5.0, 5.1, 4.9]               # GPU ~2x slower, tight
        ci = S.log_ratio_ci(gpu, cpu)
        self.assertAlmostEqual(ci["ratio"], 0.5, delta=0.01)
        self.assertLess(ci["hi"], 0.55)
        self.assertGreater(ci["lo"], 0.45)
        self.assertEqual(S.verdict(ci), "cpu_faster")

    def test_gpu_faster_and_inconclusive_and_unconfirmed(self):
        self.assertEqual(S.verdict(S.log_ratio_ci([30, 31, 29], [10, 10.3, 9.7])), "gpu_faster")
        noisy = S.log_ratio_ci([8, 12, 10], [10, 14, 7])
        self.assertEqual(S.verdict(noisy), "inconclusive")
        single = S.log_ratio_ci([30.0], [10.0])
        self.assertEqual(S.verdict(single), "unconfirmed")                # one noisy measurement never decides
        self.assertAlmostEqual(single["ratio"], 3.0)
        self.assertEqual(S.verdict(S.log_ratio_ci([], [1.0])), "no_data")

    def test_zero_variance_does_not_crash(self):
        ci = S.log_ratio_ci([2.0, 2.0], [1.0, 1.0])
        self.assertEqual(S.verdict(ci), "gpu_faster")

    def test_ci_widens_with_noise_and_fewer_samples(self):
        tight = S.log_ratio_ci([10, 10.1, 9.9, 10], [10, 10.1, 9.9, 10])
        loose = S.log_ratio_ci([7, 13, 10], [8, 12, 10])
        self.assertLess(tight["hi"] / tight["lo"], loose["hi"] / loose["lo"])

    def test_interpolation(self):
        x = S.interpolate_parity(10, 0.5, 100, 2.0)                      # symmetric in logs -> geometric midpoint
        self.assertAlmostEqual(x, math.sqrt(10 * 100), places=6)
        self.assertIsNone(S.interpolate_parity(10, 0.5, 100, 0.9))       # no straddle
        self.assertIsNone(S.interpolate_parity(10, 1.5, 100, 2.0))

    def test_crossover_only_from_confirmed_verdicts(self):
        pts = [{"x": 1, "ratio": 0.2, "verdict": "cpu_faster"}, {"x": 5, "ratio": 0.6, "verdict": "cpu_faster"},
               {"x": 20, "ratio": 1.8, "verdict": "gpu_faster"}, {"x": 50, "ratio": 3.0, "verdict": "gpu_faster"}]
        res = S.crossover_scan(pts)
        self.assertEqual(len(res["confirmed"]), 1)
        self.assertEqual(res["confirmed"][0]["between"], [5, 20])
        self.assertTrue(5 < res["confirmed"][0]["interpolated_x"] < 20)

    def test_single_measurement_crossing_is_only_a_candidate(self):
        pts = [{"x": 1, "ratio": 0.2, "verdict": "unconfirmed"}, {"x": 20, "ratio": 1.5, "verdict": "unconfirmed"}]
        res = S.crossover_scan(pts)
        self.assertEqual(res["confirmed"], [])
        self.assertEqual(len(res["candidates"]), 1)

    def test_no_crossover_when_cpu_always_faster(self):
        pts = [{"x": x, "ratio": r, "verdict": "cpu_faster"} for x, r in ((1, 0.1), (5, 0.2), (20, 0.4))]
        res = S.crossover_scan(pts)
        self.assertEqual((res["confirmed"], res["candidates"]), ([], []))

    def test_missing_data_skipped(self):
        pts = [{"x": 1, "ratio": None, "verdict": "no_data"}, {"x": 5, "ratio": 2.0, "verdict": "gpu_faster"}]
        self.assertEqual(S.crossover_scan(pts), {"confirmed": [], "candidates": []})

    def test_break_even(self):
        # GPU: 1.0 s fixed + 1 ms/step ; CPU: 0.1 s fixed + 10 ms/step  -> S* = 0.9 / 0.009 = 100
        self.assertAlmostEqual(S.break_even_steps(1.0, 0.001, 0.1, 0.010), 100.0)
        self.assertIsNone(S.break_even_steps(1.0, 0.020, 0.1, 0.010))    # GPU slower per step: never
        self.assertIsNone(S.break_even_steps(0.05, 0.001, 0.1, 0.010))   # GPU cheaper everywhere: no crossing


if __name__ == "__main__":
    unittest.main()
