import threading
import time
import unittest

from tfbench import limits as L

LIM = L.ResourceLimits()          # 6 GB extra, 2 GB floor, 80 %, 512 MB swap, 20 min


def S(t=0.0, avail=10.0, pct=50.0, swap=1.0, rss=0.5):
    return L.MemSample(t=t, avail_gb=avail, sys_pct=pct, swap_used_gb=swap, rss_gb=rss)


BASE = L.Baseline(t0=0.0, avail_gb=10.0, swap_used_gb=1.0)


class CheckLimitsTests(unittest.TestCase):
    def test_defaults_match_spec(self):
        self.assertEqual((LIM.max_extra_gb, LIM.min_avail_gb, LIM.max_sys_pct, LIM.max_swap_mb,
                          LIM.max_runtime_s), (6.0, 2.0, 80.0, 512.0, 1200.0))

    def test_ok(self):
        self.assertIsNone(L.check_limits(LIM, BASE, S(t=10, avail=8.0, pct=60.0)))

    def test_extra_memory_inclusive_boundary(self):
        self.assertIsNone(L.check_limits(LIM, BASE, S(avail=4.01)))
        self.assertIn("extra system memory", L.check_limits(LIM, BASE, S(avail=4.0)))

    def test_min_available(self):
        b = L.Baseline(0.0, 4.0, 1.0)
        self.assertIn("available memory", L.check_limits(LIM, b, S(avail=1.99, pct=10)))
        self.assertIsNone(L.check_limits(LIM, b, S(avail=2.0, pct=10)))

    def test_system_pct(self):
        self.assertIn("system memory use", L.check_limits(LIM, BASE, S(avail=9.0, pct=80.0)))
        self.assertIsNone(L.check_limits(LIM, BASE, S(avail=9.0, pct=79.9)))

    def test_swap_growth(self):
        self.assertIsNone(L.check_limits(LIM, BASE, S(swap=1.0 + 511 / 1024)))
        self.assertIn("swap grew", L.check_limits(LIM, BASE, S(swap=1.0 + 512 / 1024)))
        self.assertIsNone(L.check_limits(LIM, BASE, S(swap=0.2)))            # shrinking is fine

    def test_runtime(self):
        self.assertIn("runtime", L.check_limits(LIM, BASE, S(t=1200)))
        self.assertIsNone(L.check_limits(LIM, BASE, S(t=1199)))
        self.assertIn("runtime", L.check_limits(LIM, BASE, S(t=61), runtime_budget_s=60))

    def test_limits_validation(self):
        self.assertEqual(LIM.problems(), [])
        bad = L.ResourceLimits(max_extra_gb=0, min_avail_gb=-1, max_sys_pct=120, max_swap_mb=-5, max_runtime_s=0)
        self.assertEqual(len(bad.problems()), 5)

    def test_preflight_refuses_without_changing_limits(self):
        self.assertEqual(L.preflight(LIM, S(avail=8.0, pct=50)), [])
        probs = L.preflight(LIM, S(avail=1.5, pct=85))
        self.assertGreaterEqual(len(probs), 2)
        self.assertEqual(LIM.min_avail_gb, 2.0)              # frozen dataclass: never auto-raised
        with self.assertRaises(Exception):
            LIM.min_avail_gb = 0.1                            # type: ignore[misc]


class HeadroomTests(unittest.TestCase):
    def test_headroom_is_the_smallest_of_the_three_limits(self):
        # 16 GB machine, limits 6 GB / 2 GB floor / 80 %
        self.assertAlmostEqual(L.memory_headroom_gb(LIM, S(avail=12.0, pct=25.0), 16.0), 6.0)          # extra-memory allowance binds
        self.assertAlmostEqual(L.memory_headroom_gb(LIM, S(avail=4.0, pct=50.0), 16.0), 2.0)           # availability floor binds (4-2)
        self.assertAlmostEqual(L.memory_headroom_gb(LIM, S(avail=7.0, pct=60.0), 16.0), 3.2)           # 80 % ceiling binds
        self.assertEqual(L.memory_headroom_gb(LIM, S(avail=1.0, pct=90.0), 16.0), 0.0)                 # never negative


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class MonitorTests(unittest.TestCase):
    def make(self, samples, **kw):
        it = iter(samples)
        last = [samples[-1]]

        def sampler():
            try:
                last[0] = next(it)
            except StopIteration:
                pass
            return last[0]

        clock = FakeClock()
        return L.ResourceMonitor(LIM, sampler=sampler, clock=clock, **kw), clock

    def test_no_abort_when_within_limits(self):
        m, _ = self.make([S(), S(t=1, avail=9.5), S(t=2, avail=9.0)])
        for _ in range(3):
            m.poll_once()
        self.assertFalse(m.aborted)
        m.raise_if_aborted()

    def test_abort_sets_flag_reason_and_raises(self):
        m, _ = self.make([S(), S(), S(t=1, avail=3.0)])
        m.poll_once()
        self.assertTrue(m.aborted)
        self.assertIn("extra system memory", m.reason)
        with self.assertRaises(L.LimitExceeded) as cm:
            m.raise_if_aborted()
        self.assertIn("extra system memory", cm.exception.reason)

    def test_first_reason_is_kept(self):
        m, _ = self.make([S(), S(), S(t=1, avail=3.0), S(t=2, avail=3.0, swap=5.0)])
        m.poll_once()
        first = m.reason
        m.poll_once()
        self.assertEqual(m.reason, first)

    def test_runtime_budget(self):
        m, _ = self.make([S(), S(), S(t=100)], runtime_budget_s=50)
        m.poll_once()
        self.assertIn("runtime", m.reason)

    def test_peaks_and_window_reset(self):
        m, _ = self.make([S(), S(avail=9.0, rss=1.0), S(avail=8.0, rss=2.0, pct=60), S(avail=9.5, rss=0.7)])
        m.poll_once(); m.poll_once(); m.poll_once()
        w = m.window_peaks()
        self.assertAlmostEqual(w["max_extra_gb"], 2.0)
        self.assertAlmostEqual(w["min_avail_gb"], 8.0)
        self.assertAlmostEqual(w["max_rss_gb"], 2.0)
        m.poll_once()
        w2 = m.window_peaks()                                  # new window only sees the last sample
        self.assertAlmostEqual(w2["max_extra_gb"], 0.5)
        self.assertAlmostEqual(m.run_peaks["max_extra_gb"], 2.0)   # run-wide peak persists

    def test_hard_stop_after_grace_only_once(self):
        calls = []
        m, clock = self.make([S(), S(), S(t=1, avail=3.0)], hard_grace_s=30, on_hard_stop=calls.append)
        m.poll_once()                           # abort requested at clock=0
        clock.t = 29
        m.poll_once()
        self.assertEqual(calls, [])
        clock.t = 31
        m.poll_once(); m.poll_once()
        self.assertEqual(len(calls), 1)
        self.assertIn("extra system memory", calls[0])

    def test_acknowledged_path_has_no_hard_stop(self):
        calls = []
        m, clock = self.make([S(), S(t=1, avail=3.0)], hard_grace_s=30, on_hard_stop=calls.append)
        m.poll_once()
        self.assertTrue(m.aborted)
        # worker notices the flag and exits long before the grace period
        self.assertEqual(calls, [])

    def test_thread_start_stop_and_abort(self):
        state = {"avail": 10.0}

        def sampler():
            return S(t=time.monotonic(), avail=state["avail"])

        m = L.ResourceMonitor(LIM, interval=0.01, sampler=sampler)
        with m:
            time.sleep(0.05)
            self.assertFalse(m.aborted)
            state["avail"] = 1.0
            deadline = time.time() + 2
            while not m.aborted and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(m.aborted)
        self.assertFalse(any(t.name == "resource-monitor" and t.is_alive() for t in threading.enumerate()))

    def test_failing_sampler_does_not_kill_thread(self):
        n = {"i": 0}

        def sampler():
            n["i"] += 1
            if n["i"] > 2:
                raise RuntimeError("boom")
            return S(t=time.monotonic())

        m = L.ResourceMonitor(LIM, interval=0.01, sampler=sampler)
        m.start()
        time.sleep(0.1)
        m.stop()
        self.assertGreater(n["i"], 3)


if __name__ == "__main__":
    unittest.main()
