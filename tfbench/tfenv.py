"""TensorFlow environment helpers: device selection, placement checks, GPU sync."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


class DeviceUnavailable(RuntimeError):
    """Requested device is not present.  The suite NEVER substitutes another device."""


def import_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    return tf


def configure(device: str, placement: str, grappler: str = "default"):
    """Must run before TensorFlow initialises its runtime.

    device    'cpu' hides every GPU; 'gpu' requires one (else DeviceUnavailable)
    placement 'strict' (soft placement OFF -> an op without a GPU kernel raises) or
              'soft'   (TensorFlow may silently run unsupported ops on the CPU; the report
                        labels every measurement with placement=soft)
    """
    if device not in ("cpu", "gpu") or placement not in ("strict", "soft"):
        raise ValueError((device, placement))
    tf = import_tf()
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    tf.config.set_soft_device_placement(placement == "soft")
    if grappler == "no-remapper":                 # no MatMul+BiasAdd+activation fusion
        tf.config.optimizer.set_experimental_options({"remapper": False})
    elif grappler == "none":                      # no graph rewriting at all
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise DeviceUnavailable(
            "no GPU is visible to TensorFlow (is tensorflow-metal installed and compatible?). "
            "Refusing to fall back to the CPU silently.")
    return tf


def device_scope(tf, device: str):
    return tf.device("/GPU:0" if device == "gpu" else "/CPU:0")


_PLACEMENT_TOKENS = ("cannot assign a device", "could not satisfy explicit device", "no kernel",
                     "no supported kernel", "colocat", "not supported on gpu", "unsupported on gpu",
                     "no opkernel was registered", "device specification")


def is_placement_error(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(tok in msg for tok in _PLACEMENT_TOKENS)


class SimpleAdam:
    """Adam with the SAME update rule as Keras' legacy Adam (epsilon outside the sqrt, bias-corrected step size),
    but with only float32 state (no int64 step counter).

    Why: Keras' optimizers keep an int64 ``iterations`` variable, which tensorflow-metal cannot place on the GPU,
    forcing soft placement. This class lets the whole training step run under STRICT placement. It is used
    identically on CPU and GPU, so comparisons stay fair; use ``--optimizer keras`` for the Keras class.
    """

    def __init__(self, tf, lr: float, params, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-7):
        self.tf, self.lr, self.b1, self.b2, self.eps = tf, float(lr), beta_1, beta_2, epsilon
        self.m = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.v = [tf.Variable(tf.zeros_like(p), trainable=False) for p in params]
        self.t = tf.Variable(0.0, trainable=False)

    def variables(self):
        return [*self.m, *self.v, self.t]

    def apply_gradients(self, grads_and_vars):
        tf = self.tf
        self.t.assign_add(1.0)
        t = self.t.read_value()
        lr_t = self.lr * tf.sqrt(1.0 - tf.pow(self.b2, t)) / (1.0 - tf.pow(self.b1, t))
        for (g, p), m, v in zip(grads_and_vars, self.m, self.v):
            m_new = self.b1 * m + (1.0 - self.b1) * g
            v_new = self.b2 * v + (1.0 - self.b2) * tf.square(g)
            m.assign(m_new)
            v.assign(v_new)
            p.assign_sub(lr_t * m_new / (tf.sqrt(v_new) + self.eps))


def make_optimizer(tf, lr: float, params=(), kind: str = "simple") -> Tuple[Any, str]:
    """kind 'simple': SimpleAdam (strict-placement friendly, float32 state only).
    kind 'keras': Keras legacy Adam when it exists (documented as faster than the 2.11+ class on Apple Silicon)."""
    if kind == "simple":
        return SimpleAdam(tf, lr, list(params)), "tfbench.SimpleAdam (Keras-Adam update rule, float32 state)"
    legacy = getattr(getattr(tf.keras.optimizers, "legacy", None), "Adam", None)
    cls = legacy or tf.keras.optimizers.Adam
    return cls(learning_rate=lr), f"{cls.__module__}.{cls.__name__}"


def sync(*items) -> None:
    """Block until all queued device work feeding ``items`` has finished (host readback)."""
    for it in items:
        it.numpy()


def device_kind(dev: str) -> str:
    d = (dev or "").upper()
    return "GPU" if "GPU" in d else ("CPU" if "CPU" in d else "UNKNOWN")


def placement_evidence(tf, device: str, variables, tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Gather what TensorFlow reports about where things actually live.

    NOTE: TensorFlow exposes no public per-op *final* placement for graph functions.  The
    evidence is (1) a probe op's output device, (2) the device of every model variable,
    (3) the devices of the input/output tensors.  With soft placement OFF an op lacking a
    kernel on the requested device raises instead of moving.
    """
    with device_scope(tf, device):
        probe = tf.matmul(tf.ones((8, 8)), tf.ones((8, 8)))
        probe.numpy()
    var_devs: Dict[str, int] = {}
    for v in variables:
        k = device_kind(v.device)
        var_devs[k] = var_devs.get(k, 0) + 1
    ev = {
        "requested": device.upper(),
        "probe_op_device": probe.device,
        "variable_device_counts": var_devs,
        "tensor_devices": {k: getattr(t, "device", "n/a") for k, t in tensors.items()},
        "visible_gpus": [d.name for d in tf.config.get_visible_devices("GPU")],
        "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        "eager_default": bool(tf.executing_eagerly()),
    }
    want = device.upper()
    problems: List[str] = []
    if device_kind(probe.device) != want:
        problems.append(f"probe op ran on {probe.device}, expected {want}")
    if var_devs.get(want, 0) != sum(var_devs.values()):
        problems.append(f"model variables not all on {want}: {var_devs}")
    if device == "cpu" and ev["visible_gpus"]:
        problems.append("GPU is still visible in a CPU-only run")
    ev["verified"] = not problems
    ev["problems"] = problems
    return ev


def tf_info(tf) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "version": tf.__version__,
        "physical_devices": [f"{d.device_type}:{d.name}" for d in tf.config.list_physical_devices()],
        "visible_gpus": [d.name for d in tf.config.get_visible_devices("GPU")],
        "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "eager_by_default": bool(tf.executing_eagerly()),
        "grappler_options": {k: v for k, v in tf.config.optimizer.get_experimental_options().items()
                             if k in ("remapper", "disable_meta_optimizer", "layout_optimizer", "constant_folding")},
    }
    try:
        bi = tf.sysconfig.get_build_info()
        info["build_info"] = {k: bi[k] for k in ("is_cuda_build", "is_rocm_build") if k in bi}
    except Exception:
        pass
    try:
        info["mixed_precision_policy"] = tf.keras.mixed_precision.global_policy().name
    except Exception:
        pass
    return info


def gpu_memory_info(tf):
    """TensorFlow-reported GPU memory (usually unsupported by tensorflow-metal)."""
    try:
        m = tf.config.experimental.get_memory_info("GPU:0")
        return {"available": True, "current_gb": m["current"] / 1024 ** 3, "peak_gb": m["peak"] / 1024 ** 3}
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {str(e)[:120]}"}
