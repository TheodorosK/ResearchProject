> **EXAMPLE ONLY - not a benchmark of your Mac.** Generated in a Linux x86-64 cloud sandbox with CPU-only
> TensorFlow 2.15.1 and `--quick` settings, to show the report structure. The GPU job is listed as *unsupported*
> on purpose: that is how the suite reports a missing GPU (it never falls back to the CPU silently).
> Do not read any number below as representative of Apple Silicon.

# TensorFlow benchmark report

*Run `example_report` - started 2026-11-06T23:40:38+00:00 - finished 2026-11-06T23:41:17+00:00 - suite v2.0.0*

**Command:** `tf_memory_stress_test.py --tests performance neuroevolution memdiag --quick --ne-devices cpu --memdiag-devices cpu --out-dir docs/example_report`

**Overall status:** INCOMPLETE (1 job(s) did not complete)

**How to read numbers:** **measured** = read directly from a clock, OS counter or TensorFlow; **calculated** = derived arithmetically from measured values (median, ratio, rate, slope); **estimated** = analytical model or heuristic; NOT observed.

## 1. Hardware and software configuration

| Item | Value |
|---|---|
| Machine | Linux-6.18.44-fc-v300-x86_64-with-glibc2.39 |
| Chip | x86_64 |
| Cores (P/E, logical) | None/None, 4 |
| GPU cores | n/a |
| Unified memory (GB) | 15.71 |
| OS | Linux-6.18.44-fc-v300-x86_64-with-glibc2.39 |
| Power source | n/a |
| Low Power Mode | n/a |
| Python | 3.11.18 |
| TensorFlow / Keras | 2.15.1 / 2.15.0 |
| tensorflow-metal | n/a |
| numpy / psutil | 1.26.4 / 7.2.2 |
| Seed | 1234 |

**Safety limits:** extra memory <= 6.0 GB, available >= 2.0 GB, system use < 80.0%, swap growth < 512.0 MB, runtime <= 20 min per benchmark.

## 2. TensorFlow and device configuration

| Job | Status | Placement | Soft placement | Execution | Placement verified | Probe op device | Variable devices | Optimizer |
|---|---|---|---|---|---|---|---|---|
| performance@cpu:mlp | completed | strict | False | graph (tf.function, jit_compile=False) | True | /job:localhost/replica:0/task:0/device:CPU:0 | {'CPU': 8} | tfbench.SimpleAdam (Keras-Adam update rule, float32 state) |
| performance@gpu:mlp | unsupported | strict | n/a | n/a | n/a | n/a | None | n/a |
| neuroevolution@cpu | completed | strict | False | graph (tf.function training loop) | True | /job:localhost/replica:0/task:0/device:CPU:0 | {'CPU': 4} | tfbench.SimpleAdam (Keras-Adam update rule, float32 state) |

## 3. CPU and GPU performance (steady state)

| Workload | Device | Status | Steps/s | Samples/s | ms/step | Median run (s) | CV | Min run (s) | Max run (s) |
|---|---|---|---|---|---|---|---|---|---|
| mlp | cpu | completed | 84.54 | 10,821 | 11.83 | 0.355 | 0.0251 | 0.34 | 0.355 |
| mlp | gpu | unsupported | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_Steps/s, samples/s, ms/step, median and CV are **calculated** from the measured per-repeat times._

**CPU vs GPU (calculated):**

_no data_

_The early trajectory compares the first ~10 training-step losses on both devices. Final losses are NOT compared as a pass/fail test: after hundreds of steps the models memorise the small data pool (loss ~1e-5) and float32 summation-order differences between CPU and GPU make relative differences of such tiny numbers large and meaningless (see `final_loss_rel_diff_informational` in report.json)._

**Forward-pass diagnosis (before any weight update; errors are relative to a float64 host calculation from the same weights and batch - measured):**

| Job | Placement | Weights checksum | Batch checksum | Host-truth loss | Eager fwd err | Graph fwd err | First train-step err |
|---|---|---|---|---|---|---|---|
| performance@cpu:mlp | strict | 80,315 | 80,412 | 2.43 | 4.15e-08 | 4.15e-08 | 4.15e-08 |

_Same checksums on CPU and GPU = identical weights and data. Eager err small but graph err large = graph-mode op fusion on that device; graph fwd ok but first-step err large = the training step itself (variable update ordering etc.)._

**Phase breakdown (measured, seconds) - never merged with steady-state time:**

| Job | Data gen | Data transfer | Model build | Optimizer init | Trace | First execution | Warm-up total | Compile overhead (estimated) |
|---|---|---|---|---|---|---|---|---|
| performance@cpu:mlp | 0.0089 | 0.0246 | 0.173 | 0.0326 | 0.15 | 0.141 | 0.0636 | 0.129 |
| performance@gpu:mlp | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Thermal / stability indicators:**

| Job | pmset throttling indicated | Drift last-vs-first third (x) |
|---|---|---|
| performance@cpu:mlp | unavailable | 0.957 |
| performance@gpu:mlp | unavailable | n/a |

_`None`/`n/a` means the telemetry was unavailable, which does **not** mean no throttling occurred._

## 4. Baseline and peak memory

| Job | Peak RSS (GB) | Peak extra system memory (GB) | Min available (GB) | Max system use (%) |
|---|---|---|---|---|
| performance@cpu:mlp | 0.495 | 0.154 | 14.83 | 5.60 |
| neuroevolution@cpu | 0.409 | 0.0251 | 14.93 | 5.00 |

_Peak extra memory = drop in system-available memory vs. the job's own baseline (captures unified/GPU memory that RSS misses). All **measured** (sampled ~0.5 s)._

## 5. Swap activity

| Job | Swap growth during steady state (MB) | Page-outs during steady state (MB) |
|---|---|---|
| performance@cpu:mlp | 0 | 0 |
| performance@gpu:mlp | n/a | n/a |

| Neuroevolution job | Max swap growth since start (MB) |
|---|---|
| neuroevolution@cpu | 0 |

## 6. Neuroevolution throughput

| Job | Status | Evaluations | Failed | Individuals/s | Total (s) | Val R2 gain (gen0->last) | Best val R2 | Best test R2 (final only) | Best architecture |
|---|---|---|---|---|---|---|---|---|---|
| neuroevolution@cpu | completed | 16 | 0 | 3.79 | 4.23 | 0.188 | 0.717 | 0.696 | [128r-32r-32s] lr=1.73e-02 |

## 7. Time per generation

**neuroevolution@cpu**

| Gen | Evaluated | Failed | Wall (s) | Ind/s | Mean fit time (s) | Best val R2 | Mean val R2 | CPU cores busy | System CPU % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 0 | 1.30 | 4.62 | 0.164 | 0.529 | 0.103 | 1.06 | 28.10 |
| 1 | 5 | 0 | 1.42 | 3.52 | 0.215 | 0.589 | 0.257 | 1.03 | 27.00 |
| 2 | 5 | 0 | 1.17 | 4.26 | 0.183 | 0.717 | 0.572 | 1.07 | 27.60 |

## 8. Memory growth per generation / diagnostic

**neuroevolution@cpu**

| Gen | RSS before cleanup (GB) | RSS after (GB) | Extra system after (GB) | Peak extra (GB) | Available after (GB) | Swap since start (MB) | Cleanup (s) |
|---|---|---|---|---|---|---|---|
| 0 | 0.395 | 0.395 | 0.0251 | 0.0251 | 14.93 | 0 | 0.0977 |
| 1 | 0.404 | 0.404 | 0.0224 | 0.0233 | 14.93 | 0 | 0.0896 |
| 2 | 0.409 | 0.409 | 0.0264 | 0.0238 | 14.93 | 0 | 0.132 |

Growth per generation (calculated): system 0.691 MB, RSS 7.37 MB. Rising RSS alone is not a leak: allocator caching also grows then plateaus.

**Memory-growth diagnostic (`memdiag`, each policy in its own process; verdicts are heuristic):**

| Job | Status | RSS verdict | RSS MB/cycle | System verdict | System MB/cycle | Python MB/cycle |
|---|---|---|---|---|---|---|
| memdiag@cpu:none | completed | bounded | 0.597 | bounded | 0.0664 | 0.0975 |
| memdiag@cpu:gc | completed | bounded | 0.375 | bounded | -0.86 | 0.02 |
| memdiag@cpu:clear | completed | bounded | 0.00558 | bounded | -0.813 | 0.0175 |

**Per-cycle series, device=cpu (measured, MB; RSS / extra system memory):**

| Cycle | clear RSS | gc RSS | none RSS | clear sys | gc sys | none sys |
|---|---|---|---|---|---|---|
| 0 | 400.03 | 400.18 | 399.82 | 30.73 | 28.79 | 18.10 |
| 1 | 401.38 | 401.51 | 403.01 | 34.71 | 35.68 | 18.84 |
| 2 | 402.90 | 403.65 | 405.56 | 34.71 | 39.37 | 20.56 |
| 3 | 402.98 | 404.04 | 407.40 | 34.71 | 39.37 | 20.45 |
| 4 | 401.42 | 402.47 | 406.15 | 28.34 | 38.56 | 18.59 |
| 5 | 402.99 | 404.55 | 408.75 | 28.34 | 38.56 | 19.99 |
| 6 | 403.12 | 405.03 | 409.23 | 28.34 | 38.00 | 20.48 |
| 7 | 403.21 | 405.50 | 410.33 | 28.34 | 37.13 | 21.47 |
| 8 | 403.33 | 405.98 | 410.46 | 28.03 | 37.13 | 21.46 |
| 9 | 401.68 | 404.88 | 409.57 | 27.34 | 32.78 | 18.66 |

_`bounded`/`saturating` = consistent with allocator caching; `python_retention` = references still held; `continued_native_growth` = potential native leak, needs a longer run to confirm._

## 9. Incomplete or aborted tests

| Job | Status | Reason |
|---|---|---|
| performance@gpu:mlp | unsupported | no GPU is visible to TensorFlow (is tensorflow-metal installed and compatible?). Refusing to fall back to the CPU silently. |

## 10. Measurement limitations and notes

- Training time per individual includes per-model graph tracing/compilation (realistic for neuroevolution, where every individual is a new model).
- A fixed reference genome is evaluated once before generation 0 (not counted in the evaluation totals). It absorbs one-time start-up cost (runtime/Metal initialisation, first compilation) and its val R2 is compared across devices to detect numerical disagreement.
- Per-generation growth is a coarse indicator only; use the `memdiag` test to distinguish retained memory, allocator caching and leaks.
- Test R2 was computed once, after evolution, for the best-by-validation individual only.
- Classification is heuristic: a leak is suspected only when growth continues at a similar rate across the whole post-warm-up window; rising RSS alone is not evidence of a leak.
- RSS excludes Metal/unified-memory buffers, so `system_extra` (drop in available memory) is reported alongside it; that series is noisy while other applications run.

- Single machine, single session: the coefficient of variation (CV) captures repeat-to-repeat noise only; session-to-session variance (background load, temperature, OS state) is unknown. Repeat on separate days before trusting small differences.
- System-memory figures come from psutil (macOS 'available' memory). They are coarse: they include file cache, macOS compresses memory, and values can lag or go negative. Peaks are sampled every ~0.5 s, so shorter spikes are missed.
- Process RSS excludes Metal/unified-memory buffers; GPU memory use is therefore visible only as a drop in system-available memory. tensorflow-metal does not implement tf.config.experimental.get_memory_info.
- GPU utilisation, GPU clocks and GPU temperature are not exposed through TensorFlow; `powermetrics` needs sudo and is not used. pmset thermal fields are coarse and may be absent - absence is NOT evidence of no throttling.
- TensorFlow offers no public API for the final per-op placement of a graph function. Device placement is verified from a probe op, every model variable, and (with soft placement OFF) from the fact that an op without a GPU kernel raises instead of moving. If placement is 'soft', some ops may have run on the CPU.
- Resource limits are enforced by an in-process watchdog (cooperative: checked between steps/individuals) and by an external supervisor that SIGTERM/SIGKILLs a worker that stays over a limit. A native call blocked inside Metal cannot be interrupted from Python; the supervisor is the backstop.
- GPU floating-point reductions are not bit-reproducible: seeds fix data, initialisation and the GA lineage, but numeric results can differ slightly between runs and between CPU and GPU.
- Workloads are synthetic: a dense MLP, a small convnet and small evolved MLPs. They do not represent transformers, large convolutional models or data-pipeline-bound training. Neuroevolution models are tiny, so that benchmark is dominated by per-model Keras/TensorFlow overhead and may favour the CPU.
- FLOP/s and pre-run memory figures are analytical ESTIMATES, not measurements.
- No performance claim is made about other Apple Silicon chips or memory sizes. Comparisons require actually running this suite (same TensorFlow/tensorflow-metal versions, seeds and settings) on the other machine.
