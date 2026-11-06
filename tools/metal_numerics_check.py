#!/usr/bin/env python3
"""Op-by-op numerical check: CPU vs GPU vs a float64 NumPy reference (TensorFlow + numpy only).

When a CPU run and a Metal-GPU run starting from identical weights and data disagree, this finds WHICH op
(or which execution mode: eager vs tf.function graph, where TensorFlow may fuse ops) disagrees. Every case runs on
both devices with soft placement OFF (an op without a GPU kernel is reported as unsupported, never silently moved)
and its error is measured against float64 ground truth.

    python tools/metal_numerics_check.py
    python tools/metal_numerics_check.py --batch 128 --in-dim 784 --hidden 1024

Error = max|result - reference| / max|reference|. Float32 rounding gives ~1e-6; anything near 1e-3 or above is a
real numerical difference.
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


def glorot(rng, shape):
    limit = (6.0 / (shape[0] + shape[1])) ** 0.5
    return rng.uniform(-limit, limit, shape).astype(np.float32)


def softmax64(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--in-dim", type=int, default=784)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import tensorflow as tf
    tf.config.set_soft_device_placement(False)
    gpus = tf.config.list_physical_devices("GPU")
    devices = ["/CPU:0"] + (["/GPU:0"] if gpus else [])
    print(f"TensorFlow {tf.__version__} | GPUs visible: {len(gpus)}")
    if not gpus:
        print("No GPU visible: only CPU-vs-float64 errors will be shown.")

    rng = np.random.default_rng(a.seed)
    B, I, H, C = a.batch, a.in_dim, a.hidden, a.classes
    x = rng.standard_normal((B, I)).astype(np.float32)
    w1, w2 = glorot(rng, (I, H)), glorot(rng, (H, C))
    b1 = (rng.standard_normal(H) * 0.1).astype(np.float32)          # NON-zero biases
    b2 = (rng.standard_normal(C) * 0.1).astype(np.float32)
    labels = rng.integers(0, C, B).astype(np.int32)
    stack = rng.standard_normal((6, B, I)).astype(np.float32)
    small_a, small_b = (rng.standard_normal((8, 8)).astype(np.float32) for _ in range(2))
    act_in = (x[:, :H] if I >= H else np.tile(x, (1, H // I + 1))[:, :H]).astype(np.float32)

    x64, w164, w264 = x.astype(np.float64), w1.astype(np.float64), w2.astype(np.float64)
    b164, b264 = b1.astype(np.float64), b2.astype(np.float64)
    z1 = x64 @ w164 + b164
    h1 = np.maximum(z1, 0)
    logits64 = h1 @ w264 + b264
    p = softmax64(logits64)
    dlog = (p - np.eye(C)[labels]) / B
    dh = (dlog @ w264.T) * (z1 > 0)
    xent64 = np.array(-np.log(p[np.arange(B), labels]).mean())

    def keras_factory(with_loss):
        """Built eagerly on the target device (set_weights cannot run inside tf.function); only the call is traced."""
        def factory():
            m = tf.keras.Sequential([tf.keras.layers.Input((I,)),
                                     tf.keras.layers.Dense(H, activation="relu", kernel_initializer="zeros"),
                                     tf.keras.layers.Dense(C, kernel_initializer="zeros")])
            m.set_weights([w1, b1, w2, b2])
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if with_loss:
                return lambda x, w1, b1, w2, b2, y: loss(y, m(x, training=True))
            return lambda x, w1, b1, w2, b2, y: m(x, training=True)
        return factory

    def grad_w1(x, w1, b1, w2, b2, y):
        with tf.GradientTape() as tape:
            tape.watch(w1)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                y, tf.matmul(tf.nn.relu(tf.matmul(x, w1) + b1), w2) + b2))
        return tape.gradient(loss, w1)

    full = {"x": x, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "y": labels}
    cases = {
        "host<->device copy": ({"x": x}, lambda x: tf.identity(x) * 1.0, x.astype(np.float64)),
        "matmul 8x8": ({"a": small_a, "b": small_b}, lambda a, b: tf.matmul(a, b),
                       small_a.astype(np.float64) @ small_b.astype(np.float64)),
        f"matmul {B}x{I}@{I}x{H}": ({"x": x, "w": w1}, lambda x, w: tf.matmul(x, w), x64 @ w164),
        "dense+bias+relu (fusable)": ({"x": x, "w": w1, "b": b1}, lambda x, w, b: tf.nn.relu(tf.matmul(x, w) + b), h1),
        "matmul+bias_add": ({"x": x, "w": w1, "b": b1}, lambda x, w, b: tf.nn.bias_add(tf.matmul(x, w), b), z1),
        "matmul+bias_add+relu": ({"x": x, "w": w1, "b": b1},
                                 lambda x, w, b: tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w), b)), h1),
        "matmul+bias_add+tanh": ({"x": x, "w": w1, "b": b1},
                                 lambda x, w, b: tf.tanh(tf.nn.bias_add(tf.matmul(x, w), b)), np.tanh(z1)),
        "tanh": ({"v": act_in * 0.5}, lambda v: tf.tanh(v), np.tanh(act_in.astype(np.float64) * 0.5)),
        "sigmoid": ({"v": act_in}, lambda v: tf.sigmoid(v), 1 / (1 + np.exp(-act_in.astype(np.float64)))),
        "swish (silu)": ({"v": act_in}, lambda v: tf.nn.silu(v),
                         act_in.astype(np.float64) / (1 + np.exp(-act_in.astype(np.float64)))),
        "elu": ({"v": act_in}, lambda v: tf.nn.elu(v), np.where(act_in > 0, act_in, np.expm1(act_in.astype(np.float64)))),
        "strided slice stack[3]": ({"s": stack}, lambda s: s[3] * 1.0, stack[3].astype(np.float64)),
        "2-layer logits (raw ops)": (full, lambda x, w1, b1, w2, b2, y: tf.matmul(tf.nn.relu(tf.matmul(x, w1) + b1), w2) + b2,
                                     logits64),
        "keras Dense stack forward": (full, keras_factory(False), logits64),
        "keras SCCE loss": (full, keras_factory(True), xent64),
        "dL/dW1 (backprop)": (full, grad_w1, x64.T @ dh),
    }

    def err(out, ref):
        out = np.asarray(out, np.float64)
        return float(np.max(np.abs(out - ref)) / max(float(np.max(np.abs(ref))), 1e-30))

    def run_case(fn, inputs, dev, graph):
        with tf.device(dev):
            tens = {k: tf.constant(v) for k, v in inputs.items()}
            if getattr(fn, "__name__", "") == "factory":
                fn = fn()
            f = tf.function(fn) if graph else fn
            out = f(**tens)
            return out.numpy() if hasattr(out, "numpy") else np.asarray(out)

    suspects = []
    for graph in (False, True):
        mode = "graph (tf.function)" if graph else "eager"
        print(f"\n=== {mode} ===")
        print(f"{'case':<28}" + "".join(f"{d + ' err':>16}" for d in devices) + f"{'GPU vs CPU':>13}   verdict")
        for name, (inputs, fn, ref) in cases.items():
            res, outs = {}, {}
            for d in devices:
                try:
                    outs[d] = run_case(fn, inputs, d, graph)
                    res[d] = err(outs[d], ref)
                except Exception as e:  # noqa: BLE001
                    msg = str(e).lower()
                    res[d] = "UNSUPPORTED" if ("device" in msg or "kernel" in msg) else f"ERR:{type(e).__name__}"
            row = f"{name:<28}" + "".join(f"{(r if isinstance(r, str) else format(r, '.2e')):>16}" for r in res.values())
            verdict = ""
            if len(devices) > 1:
                if "/GPU:0" in outs and "/CPU:0" in outs:
                    gc = err(outs["/GPU:0"], outs["/CPU:0"].astype(np.float64))
                    row += f"{gc:>13.2e}"
                    if res["/GPU:0"] > max(1e-4, 50 * res["/CPU:0"]):
                        verdict = "SUSPECT: GPU differs from float64 truth"
                        suspects.append((mode, name))
                    else:
                        verdict = "ok"
                else:
                    row += f"{'-':>13}"
                    verdict = "GPU: no kernel/error"
            print(row + "   " + verdict)
    if gpus:
        print("\n=== graph mode, GPU only, by Grappler (graph optimizer) configuration ===")
        names = ["matmul+bias_add", "matmul+bias_add+relu", "keras Dense stack forward", "keras SCCE loss"]
        defaults = {"remapper": True, "disable_meta_optimizer": False}
        configs = [("default", {}), ("remapper OFF", {"remapper": False}),
                   ("all graph optimizers OFF", {"disable_meta_optimizer": True})]
        print(f"{'configuration':<28}" + "".join(f"{n[:22]:>24}" for n in names))
        for label, opts in configs:
            tf.config.optimizer.set_experimental_options({**defaults, **opts})
            row = f"{label:<28}"
            for n in names:
                inputs, fn, ref = cases[n]
                try:
                    row += f"{err(run_case(fn, inputs, '/GPU:0', True), ref):>24.2e}"
                except Exception as e:  # noqa: BLE001
                    row += f"{'ERR:' + type(e).__name__:>24}"
            print(row)
        tf.config.optimizer.set_experimental_options(defaults)
        print("If an error is ~1e-1 in 'default' but ~1e-6 with the remapper (or all optimizers) OFF, that graph "
              "optimization is what breaks the GPU result; run the suite with --grappler no-remapper | none.")
    print("\nErrors ~1e-6 are normal float32 rounding. SUSPECT = that op/mode computes something different on the GPU.")
    print(f"{len(suspects)} suspect case(s): {suspects}")
    return 1 if suspects else 0


if __name__ == "__main__":
    sys.exit(main())
