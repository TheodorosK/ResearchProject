"""TensorFlow-free workload descriptions and analytic estimates (used by --dry-run).

All FLOP / memory numbers here are ESTIMATES from closed-form formulas; they are
never presented as measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

BYTES_F32 = 4


def mlp_layer_dims(in_dim: int, width: int, depth: int, classes: int) -> List[Tuple[int, int]]:
    dims, prev = [], in_dim
    for _ in range(depth):
        dims.append((prev, width))
        prev = width
    dims.append((prev, classes))
    return dims


def mlp_params(in_dim: int, width: int, depth: int, classes: int) -> int:
    return sum(i * o + o for i, o in mlp_layer_dims(in_dim, width, depth, classes))


# Convnet: Conv3x3(32,same)-ReLU-MaxPool2 -> Conv3x3(64,same)-ReLU-MaxPool2 -> Flatten -> Dense(128) -> Dense(10)
CONV_IMG = (32, 32, 3)
CONV_CLASSES = 10


def convnet_params() -> int:
    h, w, c = CONV_IMG
    p = 3 * 3 * c * 32 + 32
    p += 3 * 3 * 32 * 64 + 64
    flat = (h // 4) * (w // 4) * 64
    p += flat * 128 + 128
    p += 128 * CONV_CLASSES + CONV_CLASSES
    return p


def convnet_fwd_flops_per_sample() -> int:
    h, w, c = CONV_IMG
    f = h * w * 32 * (3 * 3 * c) * 2
    f += (h // 2) * (w // 2) * 64 * (3 * 3 * 32) * 2
    flat = (h // 4) * (w // 4) * 64
    f += flat * 128 * 2 + 128 * CONV_CLASSES * 2
    return f


def convnet_activation_floats_per_sample() -> int:
    h, w, _ = CONV_IMG
    return h * w * 32 * 2 + (h // 2) * (w // 2) * 64 * 2 + (h // 4) * (w // 4) * 64 + 128 + CONV_CLASSES


@dataclass(frozen=True)
class PerfWorkload:
    name: str
    params: int
    train_flops_per_sample: int      # ~3x forward (forward + backward) -- ESTIMATE
    input_floats_per_sample: int
    activation_floats_per_sample: int

    def estimate(self, batch: int, pool_batches: int) -> Dict[str, float]:
        weights_and_optimizer = self.params * BYTES_F32 * 4      # weights + grads + Adam m,v
        activations = batch * self.activation_floats_per_sample * BYTES_F32 * 3
        data_pool = pool_batches * batch * (self.input_floats_per_sample + 1) * BYTES_F32
        total = weights_and_optimizer + activations + data_pool
        return {
            "params": self.params,
            "train_flops_per_step": self.train_flops_per_sample * batch,
            "model_and_optimizer_gb": weights_and_optimizer / 1024 ** 3,
            "activations_gb": activations / 1024 ** 3,
            "data_pool_gb": data_pool / 1024 ** 3,
            "total_gb": total / 1024 ** 3,
        }


def perf_workload(name: str, in_dim: int, width: int, depth: int, classes: int) -> PerfWorkload:
    if name == "mlp":
        p = mlp_params(in_dim, width, depth, classes)
        return PerfWorkload("mlp", p, 6 * p, in_dim, width * depth + classes)
    if name == "convnet":
        h, w, c = CONV_IMG
        return PerfWorkload("convnet", convnet_params(), 3 * convnet_fwd_flops_per_sample(),
                            h * w * c, convnet_activation_floats_per_sample())
    raise ValueError(f"unknown workload {name!r}")


def glorot_init(shapes, seed: int):
    """Device-independent Glorot-uniform initial weights (biases zero), as float32 numpy arrays.

    Generated on the host so CPU and GPU runs start from IDENTICAL weights and no random-number op
    ever has to run on the GPU (tensorflow-metal lacks a GPU kernel for StatelessRandomGetKeyCounter).
    ``shapes`` are the shapes of ``model.get_weights()`` (kernel, bias, kernel, bias, ...).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    for shp in shapes:
        shp = tuple(shp)
        if len(shp) == 1:
            out.append(np.zeros(shp, np.float32))
            continue
        receptive = int(np.prod(shp[:-2])) if len(shp) > 2 else 1
        fan_in, fan_out = receptive * shp[-2], receptive * shp[-1]
        limit = (6.0 / (fan_in + fan_out)) ** 0.5
        out.append(rng.uniform(-limit, limit, shp).astype(np.float32))
    return out
