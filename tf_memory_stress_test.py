#!/usr/bin/env python3
"""TensorFlow memory stress test for macOS (Apple Silicon / Intel) using synthetic data.

Install:
    pip install tensorflow psutil
    pip install tensorflow-metal      # optional, Apple Silicon GPU

Run:
    python tf_memory_stress_test.py                     # all tests, default limits
    python tf_memory_stress_test.py --max-gb 8          # stop ramps at 8 GB
    python tf_memory_stress_test.py --tests alloc train # pick tests
    python tf_memory_stress_test.py --cpu               # force CPU

Tests:
    alloc  - allocate growing tensors until the limit or OOM.
    train  - train a dense net on synthetic data, doubling batch size each round.
    leak   - repeat identical training epochs and flag steady RSS growth.

Memory is measured as the DROP in system available memory since each test began
(not process RSS): with tensorflow-metal, tensors live in unified/GPU memory that
RSS does not count. Each test runs in its own subprocess so memory is clean.
Safety: ramps stop when the drop reaches --max-gb or free memory falls below
--min-avail-gb. Close other apps first for a meaningful result.
"""
import argparse
import gc
import os
import platform
import time

import numpy as np
import psutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

PROC = psutil.Process(os.getpid())
GB = 1024 ** 3


def rss_gb():
    return PROC.memory_info().rss / GB


def sys_used_pct():
    return psutil.virtual_memory().percent


def gpu_mem_gb():
    """Device memory reported by TF (may be unsupported on Metal)."""
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info["current"] / GB, info["peak"] / GB
    except Exception:
        return None


def avail_gb():
    return psutil.virtual_memory().available / GB


BASE_AVAIL = [avail_gb()]


def set_baseline():
    BASE_AVAIL[0] = avail_gb()


def used_gb():
    """System memory consumed since the baseline (captures GPU/unified memory)."""
    return BASE_AVAIL[0] - avail_gb()


def status(tag=""):
    g = gpu_mem_gb()
    gpu = f" | GPU cur/peak {g[0]:.2f}/{g[1]:.2f} GB" if g and g[1] > 0 else ""
    print(f"  [{tag:<14}] used +{used_gb():5.2f} GB | RSS {rss_gb():5.2f} GB "
          f"| system {sys_used_pct():4.1f}% | avail {avail_gb():5.1f} GB{gpu}", flush=True)


def over_limit(args):
    return used_gb() >= args.max_gb or avail_gb() < args.min_avail_gb


def is_oom(exc):
    return isinstance(exc, (tf.errors.ResourceExhaustedError, MemoryError)) or \
        "out of memory" in str(exc).lower() or "failed to allocate" in str(exc).lower()


def cleanup():
    tf.keras.backend.clear_session()
    gc.collect()


# --------------------------------------------------------------------------- tests
def test_alloc(args):
    print("\n=== TEST: tensor allocation ramp ===")
    set_baseline()
    held, step_mb, total_mb = [], args.step_mb, 0
    try:
        while not over_limit(args):
            n = step_mb * 1024 * 1024 // 4  # float32 elements
            held.append(tf.random.uniform((n,), dtype=tf.float32))
            _ = float(tf.reduce_sum(held[-1]))  # force materialisation
            total_mb += step_mb
            if len(held) % 5 == 0:
                status(f"{total_mb / 1024:.1f} GB held")
        print(f"  Stopped at limit after holding ~{total_mb / 1024:.1f} GB of tensors.")
    except Exception as e:  # noqa: BLE001
        if is_oom(e):
            print(f"  OOM after holding ~{total_mb / 1024:.1f} GB: {type(e).__name__}")
        else:
            raise
    finally:
        held.clear()
        cleanup()
    status("after free")


def build_model(width, depth, in_dim, classes):
    layers = [tf.keras.layers.Input((in_dim,))]
    layers += [tf.keras.layers.Dense(width, activation="relu") for _ in range(depth)]
    layers.append(tf.keras.layers.Dense(classes))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def synthetic_ds(batch, in_dim, classes, steps):
    """Generate synthetic batches on the fly (no dataset held in RAM)."""
    def gen():
        rng = np.random.default_rng(0)
        for _ in range(steps):
            yield (rng.standard_normal((batch, in_dim), dtype=np.float32),
                   rng.integers(0, classes, size=(batch,), dtype=np.int32))
    sig = (tf.TensorSpec((batch, in_dim), tf.float32), tf.TensorSpec((batch,), tf.int32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(2)


def test_train(args):
    print("\n=== TEST: training with doubling batch size ===")
    set_baseline()
    batch = args.start_batch
    while not over_limit(args):
        print(f"  batch={batch} width={args.width} depth={args.depth}")
        try:
            model = build_model(args.width, args.depth, args.in_dim, args.classes)
            t = time.time()
            model.fit(synthetic_ds(batch, args.in_dim, args.classes, args.steps),
                      epochs=1, steps_per_epoch=args.steps, verbose=0)
            print(f"    ok in {time.time() - t:.1f}s")
            status(f"batch {batch}")
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                print(f"    OOM at batch {batch}: {type(e).__name__}")
                break
            raise
        finally:
            cleanup()
        batch *= 2
    else:
        print("  Stopped at memory limit.")
    status("after train")


def test_leak(args):
    print("\n=== TEST: leak detection (repeated identical epochs) ===")
    set_baseline()
    model = build_model(args.width, args.depth, args.in_dim, args.classes)
    ds = synthetic_ds(args.start_batch, args.in_dim, args.classes, args.steps)
    samples, sys_samples = [], []
    for i in range(args.leak_epochs):
        model.fit(ds, epochs=1, steps_per_epoch=args.steps, verbose=0)
        gc.collect()
        samples.append(rss_gb())
        sys_samples.append(used_gb())
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
    cleanup()


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--tests", nargs="+", choices=["alloc", "train", "leak"],
                   default=["alloc", "train", "leak"])
    p.add_argument("--max-gb", type=float, default=min(8.0, psutil.virtual_memory().total / GB * 0.5),
                   help="stop ramps when this much extra system memory is used (default: min(8, 50%% RAM))")
    p.add_argument("--min-avail-gb", type=float, default=1.5,
                   help="stop ramps when available system memory falls below this")
    p.add_argument("--cpu", action="store_true", help="hide GPU (Metal) and run on CPU")
    p.add_argument("--step-mb", type=int, default=256, help="alloc test chunk size")
    p.add_argument("--start-batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=20, help="steps per epoch")
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--in-dim", type=int, default=1024)
    p.add_argument("--classes", type=int, default=100)
    p.add_argument("--leak-epochs", type=int, default=20)
    p.add_argument("--leak-mb-per-epoch", type=float, default=5.0)
    args = p.parse_args()

    if len(args.tests) > 1:  # run each test in a fresh process
        import subprocess
        import sys
        rest, skip = [], False
        for a in sys.argv[1:]:
            if a == "--tests":
                skip = True
                continue
            if skip and not a.startswith("--"):
                continue
            skip = False
            rest.append(a)
        for name in args.tests:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--tests", name, *rest])
        return

    if args.cpu:
        tf.config.set_visible_devices([], "GPU")

    print(f"Python {platform.python_version()} | TF {tf.__version__} | {platform.platform()} "
          f"({platform.machine()})")
    print(f"Devices: {[d.name for d in tf.config.list_physical_devices()]}")
    print(f"RAM total {psutil.virtual_memory().total / GB:.1f} GB | "
          f"limits: +{args.max_gb:.1f} GB used, >= {args.min_avail_gb:.1f} GB free")
    status("start")

    for name in args.tests:
        globals()[f"test_{name}"](args)

    print("\nDone.")
    status("end")


if __name__ == "__main__":
    main()
