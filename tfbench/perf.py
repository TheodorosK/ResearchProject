"""Controlled CPU-vs-GPU training benchmark (one process = one device x one workload).

Phases are timed SEPARATELY and never merged into one number:
  data_generation  numpy synthetic data (host)
  data_transfer    tf.constant placement on the target device (+ sync)
  model_build      Keras model construction
  optimizer_init   optimizer object creation
  trace            tf.function tracing (graph mode only)
  first_execution  first call: graph optimisation / kernel+shader compilation + one step (+ sync)
  warmup           W untimed-for-statistics steps (+ sync), timed for the record
  steady           R repeats x N steps, each repeat closed by a host readback (sync)

The timed loop contains no Keras callbacks and no OS queries: only an Event-flag check
(``monitor.aborted``).  Memory/thermal probes run BETWEEN repeats, outside the clock.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict

import numpy as np

from . import tfenv
from .common import Recorder, timing_stats
from .limits import LimitExceeded, ResourceMonitor, take_sample, swap_growth_mb, extra_gb
from .sysinfo import (memory_pressure_free_pct, peak_rss_gb, thermal_state, power_source, low_power_mode,
                      swap_counters)
from .workloads import CONV_CLASSES, CONV_IMG, perf_workload

pc = time.perf_counter


def _teacher_labels(rng: np.random.Generator, x2d: np.ndarray, classes: int) -> np.ndarray:
    w = rng.standard_normal((x2d.shape[1], classes)).astype(np.float32)
    return np.argmax(x2d @ w, axis=1).astype(np.int32)


def make_data(workload: str, args, seed: int):
    """Deterministic synthetic data pool: ``perf_pool`` batches that are cycled."""
    rng = np.random.default_rng(seed)
    n = args.perf_batch * args.perf_pool
    if workload == "mlp":
        x = rng.standard_normal((n, args.perf_in_dim)).astype(np.float32)
        y = _teacher_labels(rng, x, args.perf_classes)
    else:
        x = rng.standard_normal((n, *CONV_IMG)).astype(np.float32)
        y = _teacher_labels(rng, x.reshape(n, -1), CONV_CLASSES)
    return x, y


def build_model(tf, workload: str, args, seed: int):
    """Layers are created with the 'zeros' initializer (a GPU-supported op) and then loaded with
    host-generated Glorot weights, so initial weights are identical on every device."""
    from .workloads import glorot_init
    K = tf.keras.layers
    z = "zeros"
    if workload == "mlp":
        layers = [K.Input((args.perf_in_dim,))]
        layers += [K.Dense(args.perf_width, activation="relu", kernel_initializer=z) for _ in range(args.perf_depth)]
        layers.append(K.Dense(args.perf_classes, kernel_initializer=z))
    else:
        layers = [
            K.Input(CONV_IMG),
            K.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer=z),
            K.MaxPool2D(2),
            K.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer=z),
            K.MaxPool2D(2),
            K.Flatten(),
            K.Dense(128, activation="relu", kernel_initializer=z),
            K.Dense(CONV_CLASSES, kernel_initializer=z),
        ]
    model = tf.keras.Sequential(layers)
    model.set_weights(glorot_init([w.shape for w in model.get_weights()], seed))
    return model


def run(args, rec: Recorder, monitor: ResourceMonitor, device: str, workload: str, placement: str) -> None:
    tf = tfenv.configure(device, placement, args.grappler)
    seed = args.seed
    ctx = dict(device=device, workload=workload, placement=placement)
    rec.data["tf"] = tfenv.tf_info(tf)
    rec.data["config"] = {
        "device": device, "workload": workload, "placement": placement,
        "batch": args.perf_batch, "steps_per_repeat": args.perf_steps, "warmup_steps": args.perf_warmup,
        "repeats": args.perf_repeats, "pool_batches": args.perf_pool, "seed": seed,
        "mode": args.perf_mode, "optimizer": args.optimizer, "grappler": args.grappler, "dtype": "float32", "jit_compile": False,
        "width": args.perf_width, "depth": args.perf_depth,
    }
    power, lpm = power_source(), low_power_mode()
    rec.data["power_source"], rec.data["low_power_mode"] = power, lpm
    if power.get("available") and power.get("source") and "AC" not in power["source"]:
        rec.note("Machine was NOT on AC power during this job; results are not comparable to plugged-in runs.")
    if lpm.get("enabled"):
        rec.note("Low Power Mode was enabled; performance is deliberately reduced.")
    thermal = [{"when": "before_job", **thermal_state()}]
    mp_before = memory_pressure_free_pct()

    mode = args.perf_mode
    tf.keras.utils.set_random_seed(seed)

    with tfenv.device_scope(tf, device):
        # ---------------------------------------------------------------- data
        t = pc(); x_np, y_np = make_data(workload, args, seed); data_gen_s = pc() - t
        P, B = args.perf_pool, args.perf_batch
        t = pc()
        xs = [tf.constant(x_np[i * B:(i + 1) * B]) for i in range(P)]
        ys = [tf.constant(y_np[i * B:(i + 1) * B]) for i in range(P)]
        tfenv.sync(xs[-1], ys[-1], tf.reduce_sum(xs[0]))
        data_xfer_s = pc() - t
        del x_np, y_np

        # --------------------------------------------------------------- model
        t = pc(); model = build_model(tf, workload, args, seed); model_build_s = pc() - t
        tvars = model.trainable_variables
        t = pc(); opt, opt_name = tfenv.make_optimizer(tf, 1e-3, tvars, args.optimizer); optimizer_init_s = pc() - t
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        probe_var = tvars[-1]                      # tiny (last bias) readback target

        def train_step(x, y):
            with tf.GradientTape() as tape:
                loss = loss_fn(y, model(x, training=True))
            grads = tape.gradient(loss, tvars)
            opt.apply_gradients(zip(grads, tvars))
            return loss

        step_fn = train_step if mode == "eager" else tf.function(train_step, jit_compile=False)
        rec.data["execution_mode"] = ("eager (python-level op dispatch)" if mode == "eager"
                                      else "graph (tf.function, jit_compile=False)")

        # ------------------------------------- forward-pass diagnosis (before ANY update)
        from .workloads import host_forward, host_xent, weights_checksum
        w0 = model.get_weights()                       # read back from the device
        x0, y0 = xs[0].numpy(), ys[0].numpy()
        diag = {"init_weights_checksum": weights_checksum(w0),
                "batch0_checksum": float(np.abs(x0.astype(np.float64)).sum()), "batch0_label_sum": int(y0.sum()),
                "eager_forward_loss": float(loss_fn(ys[0], model(xs[0], training=False)).numpy())}
        graph_fwd = tf.function(lambda x, y: loss_fn(y, model(x, training=False)), jit_compile=False)
        diag["graph_forward_loss"] = float(graph_fwd(xs[0], ys[0]).numpy())
        if workload == "mlp":
            lg = host_forward(w0, x0, ["relu"] * args.perf_depth)
            diag["host_truth_loss"] = host_xent(lg, y0) if lg is not None else None
        rec.data["forward_diagnosis"] = diag
        for k, v in diag.items():
            rec.measure(f"diag_{k}", v, "", **ctx)
        from .workloads import numeric_gate
        gate = None if args.skip_numeric_gate else numeric_gate(diag)
        if workload != "mlp":
            rec.note(f"[{device}/{workload}] no float64 host reference exists for this workload: forward-pass numerics "
                     "were NOT validated; rely on the CPU/GPU first-step-loss comparison.")
        if gate:
            rec.set_status("failed", gate)
            rec.note("This is a numerical-correctness failure of the device/graph execution, not a benchmark failure. "
                     "Run tools/metal_numerics_check.py; on tensorflow-metal, `--grappler none` (disable all graph rewriting, applied to CPU and GPU jobs alike) is the known workaround.")
            return

        # --------------------------------------------------- trace / first step
        trace_s = None
        if mode == "graph":
            t = pc(); step_fn.get_concrete_function(xs[0], ys[0]); trace_s = pc() - t
        t = pc(); loss0 = step_fn(xs[0], ys[0]); tfenv.sync(loss0, probe_var); first_exec_s = pc() - t
        loss_first = float(loss0.numpy())

        ev = tfenv.placement_evidence(tf, device, tvars, {"input_x": xs[0], "loss": loss0})
        ev["optimizer_class"] = opt_name
        try:
            ev["optimizer_variable_device_kinds"] = sorted({tfenv.device_kind(v.device) for v in opt.variables()})
        except Exception:
            pass
        rec.data["device_evidence"] = ev
        if not ev["verified"]:
            rec.set_status("failed", "device placement could not be verified: " + "; ".join(ev["problems"]))
            rec.note("Timings are NOT reported for a run whose device placement was not confirmed.")
            return

        # --------------------------------------------------------------- warm-up
        t = pc()
        for i in range(args.perf_warmup):
            if monitor.aborted:
                raise LimitExceeded(monitor.reason)
            loss = step_fn(xs[(i + 1) % P], ys[(i + 1) % P])
        tfenv.sync(loss, probe_var) if args.perf_warmup else None
        warmup_s = pc() - t

        rec.measure("data_generation_s", data_gen_s, "s", **ctx)
        rec.measure("data_transfer_s", data_xfer_s, "s", **ctx)
        rec.measure("model_build_s", model_build_s, "s", **ctx)
        rec.measure("optimizer_init_s", optimizer_init_s, "s", **ctx)
        if trace_s is not None:
            rec.measure("trace_s", trace_s, "s", **ctx)
        rec.measure("first_execution_s", first_exec_s, "s", **ctx)
        rec.measure("warmup_total_s", warmup_s, "s", **ctx)
        rec.measure("loss_first_step", loss_first, "", **ctx)
        rec.checkpoint()

        # ------------------------------------------------------------ steady state
        N, R = args.perf_steps, args.perf_repeats
        run_times, loss_last = [], float("nan")
        base_swap = swap_counters()
        mem0 = take_sample()
        for r in range(R):
            if monitor.aborted:
                raise LimitExceeded(monitor.reason)
            t0 = pc()
            for i in range(N):
                if monitor.aborted:               # Event.is_set(): negligible cost
                    raise LimitExceeded(monitor.reason)
                loss = step_fn(xs[i % P], ys[i % P])
            tfenv.sync(loss, probe_var)           # host readback = completion barrier
            dt = pc() - t0
            run_times.append(dt)
            loss_last = float(loss.numpy())
            # ---- everything below is OUTSIDE the timed region
            s = take_sample()
            rec.measure("steady_run_s", dt, "s", repeat=r, **ctx)
            rec.measure("steady_run_samples_per_s", N * B / dt, "samples/s", kind="calculated", repeat=r, **ctx)
            rec.measure("steady_run_extra_system_gb", extra_gb(monitor.baseline, s), "GB", repeat=r, **ctx)
            th = thermal_state()
            thermal.append({"when": f"after_repeat_{r}", **th})
            rec.checkpoint()

        thermal.append({"when": "after_job", **thermal_state()})
        sw = swap_counters()

    # ------------------------------------------------------------- aggregates
    st = timing_stats(run_times)
    med = st["median"]
    for k in ("median", "mean", "stdev", "cv", "min", "max"):
        rec.measure(f"steady_run_s_{k}", st[k], "s" if k != "cv" else "ratio", kind="calculated", **ctx)
    rec.measure("steady_total_timed_s", sum(run_times), "s", kind="calculated", **ctx)
    rec.measure("steady_steps_per_s", N / med, "steps/s", kind="calculated", **ctx)
    rec.measure("steady_samples_per_s", N * B / med, "samples/s", kind="calculated", **ctx)
    sps = timing_stats([N * B / t for t in run_times])
    rec.measure("steady_samples_per_s_cv", sps["cv"], "ratio", kind="calculated", **ctx)
    rec.measure("steady_step_time_ms", med / N * 1000, "ms", kind="calculated", **ctx)
    rec.measure("compile_overhead_estimate_s", max(0.0, first_exec_s - med / N), "s", kind="estimated", **ctx)
    rec.measure("loss_final", loss_last, "", **ctx)
    rec.measure("loss_finite", bool(math.isfinite(loss_last) and math.isfinite(loss_first)), "", **ctx)

    flops = perf_workload(workload, args.perf_in_dim, args.perf_width, args.perf_depth,
                          args.perf_classes).train_flops_per_sample
    rec.measure("achieved_gflops_estimate", flops * B * N / med / 1e9, "GFLOP/s", kind="estimated", **ctx)

    prss = peak_rss_gb()
    rec.measure("peak_rss_gb", prss, "GB", **ctx)
    pk = monitor.run_peaks
    rec.measure("peak_extra_system_gb_sampled", pk["max_extra_gb"], "GB", **ctx)
    rec.measure("min_available_gb_sampled", pk["min_avail_gb"], "GB", **ctx)
    rec.measure("max_system_pct_sampled", pk["max_sys_pct"], "%", **ctx)
    swap_mb = (sw["used_gb"] - base_swap["used_gb"]) * 1024
    rec.measure("swap_growth_during_steady_mb", swap_mb, "MB", **ctx)
    rec.measure("swap_pageouts_during_steady_mb", (sw["sout_gb"] - base_swap["sout_gb"]) * 1024, "MB", **ctx)
    rec.measure("extra_system_gb_end", extra_gb(monitor.baseline, take_sample()), "GB", **ctx)
    rec.measure("memory_free_pct_before_macos", mp_before, "%", **ctx)
    rec.measure("memory_free_pct_after_macos", memory_pressure_free_pct(), "%", **ctx)

    # thermal / drift indicators ---------------------------------------------------
    rec.data["thermal_samples"] = thermal
    indicated = [x.get("throttling_indicated") for x in thermal]
    rec.measure("thermal_throttling_indicated_by_pmset",
                (True if any(i is True for i in indicated) else (False if all(i is False for i in indicated) else None)),
                "", **ctx)
    if R >= 3:
        k = max(1, R // 3)
        drift = (sorted(run_times[-k:])[len(run_times[-k:]) // 2]) / (sorted(run_times[:k])[len(run_times[:k]) // 2])
        rec.measure("steady_drift_ratio_last_vs_first_third", drift, "ratio", kind="calculated", **ctx)
        if drift > 1.10:
            rec.note(f"[{device}/{workload}] later repeats were {100 * (drift - 1):.0f}% slower than early ones: "
                     "possible thermal throttling or background load (inferred indicator, not confirmed).")
    gm = tfenv.gpu_memory_info(tf) if device == "gpu" else {"available": False, "reason": "cpu run"}
    rec.data["gpu_memory_info"] = gm
    if device == "gpu" and not gm["available"]:
        rec.note("GPU memory/utilisation are not exposed by tensorflow-metal through the TensorFlow API; "
                 "GPU memory use is therefore only visible as system-memory growth (unified memory).")
