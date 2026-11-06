"""Legacy tests ``alloc``, ``train`` and ``leak`` (CLI-compatible with the original script).

Behaviour is preserved (same loops, same printed output).  What changed:
  * limits come from ResourceLimits (defaults tightened: 6 GB extra, 2 GB floor, 80 %, 512 MB swap);
  * a few headline numbers are also recorded for the JSON/CSV/Markdown report;
  * TensorFlow is imported lazily so the supervisor process stays TensorFlow-free.
"""
from __future__ import annotations

import gc
import time
from typing import Optional

import numpy as np

from . import tfenv
from .common import Recorder
from .limits import (GB, ResourceLimits, capture_baseline, check_limits, extra_gb, swap_growth_mb,
                     take_sample)

_S = {"limits": None, "baseline": None, "budget": None, "tf": None}


def init(limits: ResourceLimits, budget_s: Optional[float], tf) -> None:
    _S.update(limits=limits, budget=budget_s, tf=tf)
    set_baseline()


def set_baseline() -> None:
    _S["baseline"] = capture_baseline()


def used_gb() -> float:
    return extra_gb(_S["baseline"], take_sample())


def over_limit(args=None) -> Optional[str]:
    return check_limits(_S["limits"], _S["baseline"], take_sample(), _S["budget"])


def status(tag: str = "") -> None:
    s = take_sample()
    print(f"  [{tag:<14}] used +{extra_gb(_S['baseline'], s):5.2f} GB | RSS {s.rss_gb:5.2f} GB "
          f"| system {s.sys_pct:4.1f}% | avail {s.avail_gb:5.1f} GB "
          f"| swap +{swap_growth_mb(_S['baseline'], s) / 1024:4.2f} GB", flush=True)


def is_oom(exc) -> bool:
    tf = _S["tf"]
    return isinstance(exc, (tf.errors.ResourceExhaustedError, MemoryError)) or \
        "out of memory" in str(exc).lower() or "failed to allocate" in str(exc).lower()


def cleanup() -> None:
    _S["tf"].keras.backend.clear_session()
    gc.collect()


# --------------------------------------------------------------------------- alloc
def test_alloc(args, rec: Recorder) -> None:
    tf = _S["tf"]
    print("\n=== TEST: tensor allocation ramp ===")
    set_baseline()
    held, step_mb, total_mb, reason = [], args.step_mb, 0, None
    try:
        # Metal commits memory lazily, so also cap by bytes we allocate ourselves.
        while True:
            reason = over_limit(args)
            if reason or total_mb / 1024 >= args.max_gb:
                reason = reason or "allocation cap (--max-gb) reached"
                break
            n = step_mb * 1024 * 1024 // 4
            held.append(tf.random.uniform((n,), dtype=tf.float32))
            _ = float(tf.reduce_sum(held[-1]))
            total_mb += step_mb
            if len(held) % 5 == 0:
                status(f"{total_mb / 1024:.1f} GB held")
        print(f"  Stopped at limit after holding ~{total_mb / 1024:.1f} GB of tensors ({reason}).")
    except Exception as e:  # noqa: BLE001
        if is_oom(e):
            reason = f"OOM: {type(e).__name__}"
            print(f"  OOM after holding ~{total_mb / 1024:.1f} GB: {type(e).__name__}")
        else:
            raise
    finally:
        held.clear()
        cleanup()
    rec.measure("alloc_requested_gb", total_mb / 1024, "GB")
    rec.measure("alloc_stop_reason", reason, "")
    rec.note("alloc only confirms TensorFlow accepted the allocation requests: with tensorflow-metal the "
             "memory is committed lazily and is not visible in system counters (use `train`/`performance`).")
    status("after free")


# --------------------------------------------------------------------------- train
def build_model(width, depth, in_dim, classes):
    tf = _S["tf"]
    layers = [tf.keras.layers.Input((in_dim,))]
    layers += [tf.keras.layers.Dense(width, activation="relu") for _ in range(depth)]
    layers.append(tf.keras.layers.Dense(classes))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def synthetic_ds(batch, in_dim, classes, steps):
    tf = _S["tf"]

    def gen():
        rng = np.random.default_rng(0)
        for _ in range(steps):
            yield (rng.standard_normal((batch, in_dim), dtype=np.float32),
                   rng.integers(0, classes, size=(batch,), dtype=np.int32))
    sig = (tf.TensorSpec((batch, in_dim), tf.float32), tf.TensorSpec((batch,), tf.int32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def _make_guard(args):
    tf = _S["tf"]
    limits = _S["limits"]

    class Guard(tf.keras.callbacks.Callback):
        """Stop mid-round on memory/swap limits (macOS swaps instead of raising OOM)."""

        def __init__(self):
            super().__init__()
            self.reason = None

        def on_train_batch_end(self, batch, logs=None):
            self.reason = over_limit(args)
            if self.reason:
                self.model.stop_training = True
    return Guard()


def test_train(args, rec: Recorder) -> None:
    print("\n=== TEST: training with doubling batch size ===")
    set_baseline()
    batch, best_sps, last_ok, peak_used, reason = args.start_batch, 0.0, None, 0.0, None
    while batch <= args.max_batch:
        print(f"  batch={batch} width={args.width} depth={args.depth}")
        guard = _make_guard(args)
        try:
            model = build_model(args.width, args.depth, args.in_dim, args.classes)
            t = time.time()
            model.fit(synthetic_ds(batch, args.in_dim, args.classes, args.steps),
                      epochs=1, steps_per_epoch=args.steps, verbose=0, callbacks=[guard])
            dt = time.time() - t
            peak_used = max(peak_used, used_gb())
            if guard.reason:
                reason = f"{guard.reason} at batch {batch}"
                status(f"batch {batch}")
                break
            sps = batch * args.steps / dt
            print(f"    ok in {dt:.1f}s ({sps:,.0f} samples/s)")
            status(f"batch {batch}")
            last_ok = batch
            if best_sps and sps < best_sps / args.slowdown:
                reason = (f"throughput fell to {sps / best_sps:.0%} of best at batch {batch} "
                          "(likely swapping)")
                break
            best_sps = max(best_sps, sps)
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                reason = f"OOM at batch {batch}: {type(e).__name__}"
                break
            raise
        finally:
            cleanup()
        batch *= 2
    print(f"  Largest batch completed: {last_ok} | peak used +{peak_used:.2f} GB | "
          f"stopped: {reason or 'reached --max-batch'}")
    rec.measure("train_largest_batch_completed", last_ok, "samples")
    rec.measure("train_peak_used_gb", peak_used, "GB")
    rec.measure("train_best_samples_per_s", best_sps, "samples/s")
    rec.measure("train_stop_reason", reason or "reached --max-batch", "")
    rec.note("`train` uses a per-batch Keras callback that forces a GPU sync; its throughput is a "
             "stress-test figure, not a clean performance measurement (use `performance`).")
    status("after train")


# --------------------------------------------------------------------------- leak
def test_leak(args, rec: Recorder) -> None:
    print("\n=== TEST: leak detection (repeated identical epochs) ===")
    set_baseline()
    model = build_model(args.width, args.depth, args.in_dim, args.classes)
    ds = synthetic_ds(args.start_batch, args.in_dim, args.classes, args.steps)
    samples, sys_samples = [], []
    for i in range(args.leak_epochs):
        model.fit(ds, epochs=1, steps_per_epoch=args.steps, verbose=0)
        gc.collect()
        s = take_sample()
        samples.append(s.rss_gb)
        sys_samples.append(extra_gb(_S["baseline"], s))
        status(f"epoch {i + 1}")
        if over_limit(args):
            print("  Memory limit hit; aborting.")
            break
    if len(samples) >= 6:
        half = len(samples) // 2
        x = np.arange(len(samples) - half)
        rss_mb = np.polyfit(x, samples[half:], 1)[0] * 1024
        sys_mb = np.polyfit(x, sys_samples[half:], 1)[0] * 1024
        verdict = "POSSIBLE LEAK" if max(rss_mb, sys_mb) > args.leak_mb_per_epoch else "stable"
        print(f"  Slope over 2nd half: RSS {rss_mb:+.1f} MB/epoch, system {sys_mb:+.1f} MB/epoch "
              f"-> {verdict}")
        print("  (system slope is noisy if other apps are running)")
        rec.measure("leak_rss_slope_mb_per_epoch", float(rss_mb), "MB/epoch", kind="calculated")
        rec.measure("leak_system_slope_mb_per_epoch", float(sys_mb), "MB/epoch", kind="calculated")
        rec.measure("leak_verdict", verdict, "", kind="calculated")
    cleanup()
