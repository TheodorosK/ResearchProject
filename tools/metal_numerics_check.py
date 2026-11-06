#!/usr/bin/env python3
"""Op-by-op numerical check: CPU vs GPU vs a float64 NumPy reference (TensorFlow + numpy only).

Purpose: when a CPU run and a Metal-GPU run starting from identical weights and data disagree, find WHICH op
disagrees. Each case runs on both devices with soft placement OFF (an op without a GPU kernel is reported as
unsupported, never silently moved) and its error is measured against float64 ground truth, so you can see
which device is closer to the truth.

    python tools/metal_numerics_check.py            # default sizes
    python tools/metal_numerics_check.py --batch 128 --in-dim 784 --hidden 1024

Error = max|result - reference| / max|reference|.  Float32 rounding gives ~1e-6..1e-5; anything near 1e-3 or
above is a real numerical difference.
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
    b1 = np.zeros(H, np.float32)
    labels = rng.integers(0, C, B).astype(np.int32)
    t = rng.standard_normal(B).astype(np.float32)
    small_a, small_b = (rng.standard_normal((8, 8)).astype(np.float32) for _ in range(2))
    act_in = (x[:, :H] if I >= H else np.tile(x, (1, H // I + 1))[:, :H]).astype(np.float32)

    x64, w164, w264 = x.astype(np.float64), w1.astype(np.float64), w2.astype(np.float64)
    z1 = x64 @ w164
    h1 = np.maximum(z1, 0)
    logits64 = h1 @ w264
    p = softmax64(logits64)
    onehot = np.eye(C)[labels]
    dlog = (p - onehot) / B
    dh = (dlog @ w264.T) * (z1 > 0)

    # name: (inputs dict, tf function, float64 reference)
    cases = {
        "host->device->host copy": ({"x": x}, lambda x: tf.identity(x) * 1.0, x.astype(np.float64)),
        "matmul 8x8": ({"a": small_a, "b": small_b}, lambda a, b: tf.matmul(a, b),
                       small_a.astype(np.float64) @ small_b.astype(np.float64)),
        f"matmul {B}x{I} @ {I}x{H}": ({"x": x, "w": w1}, lambda x, w: tf.matmul(x, w), z1),
        "bias add + relu": ({"x": x, "w": w1, "b": b1}, lambda x, w, b: tf.nn.relu(tf.matmul(x, w) + b), h1),
        "tanh": ({"v": act_in * 0.5}, lambda v: tf.tanh(v), np.tanh(act_in.astype(np.float64) * 0.5)),
        "sigmoid": ({"v": act_in}, lambda v: tf.sigmoid(v), 1 / (1 + np.exp(-act_in.astype(np.float64)))),
        "swish (silu)": ({"v": act_in}, lambda v: tf.nn.silu(v),
                         act_in.astype(np.float64) / (1 + np.exp(-act_in.astype(np.float64)))),
        "elu": ({"v": act_in}, lambda v: tf.nn.elu(v),
                np.where(act_in > 0, act_in, np.expm1(act_in.astype(np.float64)))),
        "2-layer logits": ({"x": x, "w1": w1, "w2": w2}, lambda x, w1, w2: tf.matmul(tf.nn.relu(tf.matmul(x, w1)), w2),
                           logits64),
        "sparse softmax xent": ({"lg": logits64.astype(np.float32), "y": labels},
                                lambda lg, y: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, lg)),
                                np.array(-np.log(p[np.arange(B), labels]).mean())),
        "mse": ({"pred": logits64[:, 0].astype(np.float32), "t": t},
                lambda pred, t: tf.reduce_mean(tf.square(t - pred)),
                np.array(((t.astype(np.float64) - logits64[:, 0]) ** 2).mean())),
    }

    def grad_w1(x, w1, w2, y):
        w1v = tf.Variable(w1)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                y, tf.matmul(tf.nn.relu(tf.matmul(x, w1v)), w2)))
        return tape.gradient(loss, w1v)
    cases["dL/dW1 (backprop)"] = ({"x": x, "w1": w1, "w2": w2, "y": labels}, grad_w1, x64.T @ dh)

    def err(out, ref):
        out = np.asarray(out, np.float64)
        return float(np.max(np.abs(out - ref)) / max(float(np.max(np.abs(ref))), 1e-30))

    print(f"\n{'case':<28}" + "".join(f"{d + ' err':>16}" for d in devices) + f"{'GPU vs CPU':>14}   verdict")
    suspects = 0
    for name, (inputs, fn, ref) in cases.items():
        res, outs = {}, {}
        for d in devices:
            try:
                with tf.device(d):
                    tens = {k: tf.constant(v) for k, v in inputs.items()}
                    out = fn(**tens)
                    out = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
                outs[d], res[d] = out, err(out, ref)
            except Exception as e:  # noqa: BLE001
                res[d] = "UNSUPPORTED" if "device" in str(e).lower() or "kernel" in str(e).lower() else f"ERR:{type(e).__name__}"
        row = f"{name:<28}" + "".join(f"{(r if isinstance(r, str) else format(r, '.2e')):>16}" for r in res.values())
        verdict = ""
        if len(devices) > 1:
            if "/GPU:0" in outs and "/CPU:0" in outs:
                gc = err(outs["/GPU:0"], outs["/CPU:0"].astype(np.float64))
                row += f"{gc:>14.2e}"
                cpu_e, gpu_e = res["/CPU:0"], res["/GPU:0"]
                if gpu_e > max(1e-4, 50 * cpu_e):
                    verdict, suspects = "SUSPECT: GPU result differs from float64 truth", suspects + 1
                else:
                    verdict = "ok"
            else:
                row += f"{'-':>14}"
                verdict = "GPU: no kernel/err (soft placement would silently run this on CPU)"
        print(row + "   " + verdict)
    print("\nInterpretation: errors ~1e-6..1e-5 are normal float32 rounding. A GPU error orders of magnitude above the CPU's "
          "(flagged SUSPECT) means that op computes something different on the GPU.")
    print(f"{suspects} suspect case(s).")
    return 1 if suspects else 0


if __name__ == "__main__":
    sys.exit(main())
