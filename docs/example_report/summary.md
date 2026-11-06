> **EXAMPLE ONLY - not a benchmark of your Mac.** Generated in a Linux x86-64 cloud sandbox with CPU-only
> TensorFlow 2.15.1 and `--quick` settings, to show the report structure. The GPU job is listed as *unsupported*
> on purpose: that is how the suite reports a missing GPU (it never falls back to the CPU silently).
> Do not read any number below as representative of Apple Silicon.

# TensorFlow benchmark report

*Run `example_report` - started 2026-11-06T22:35:18+00:00 - finished 2026-11-06T22:35:58+00:00 - suite v2.0.0*

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
| performance@cpu:mlp | completed | strict | False | graph (tf.function, jit_compile=False) | True | /job:localhost/replica:0/task:0/device:CPU:0 | {'CPU': 8} | keras.src.optimizers.legacy.adam.Adam |
| performance@gpu:mlp | unsupported | strict | n/a | n/a | n/a | n/a | None | n/a |
| neuroevolution@cpu | completed | strict | False | graph (Keras model.fit) | True | /job:localhost/replica:0/task:0/device:CPU:0 | {'CPU': 4} | keras.src.optimizers.legacy.adam.Adam |

## 3. CPU and GPU performance (steady state)

| Workload | Device | Status | Steps/s | Samples/s | ms/step | Median run (s) | CV | Min run (s) | Max run (s) |
|---|---|---|---|---|---|---|---|---|---|
| mlp | cpu | completed | 106.13 | 13,584 | 9.42 | 0.283 | 0.0505 | 0.262 | 0.289 |
| mlp | gpu | unsupported | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_Steps/s, samples/s, ms/step, median and CV are **calculated** from the measured per-repeat times._

**CPU vs GPU (calculated):**

_no data_

**Phase breakdown (measured, seconds) - never merged with steady-state time:**

| Job | Data gen | Data transfer | Model build | Optimizer init | Trace | First execution | Warm-up total | Compile overhead (estimated) |
|---|---|---|---|---|---|---|---|---|
| performance@cpu:mlp | 0.00745 | 0.0201 | 0.0686 | 8.13e-05 | 0.199 | 0.0785 | 0.0597 | 0.0691 |
| performance@gpu:mlp | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Thermal / stability indicators:**

| Job | pmset throttling indicated | Drift last-vs-first third (x) |
|---|---|---|
| performance@cpu:mlp | unavailable | 0.929 |
| performance@gpu:mlp | unavailable | n/a |

_`None`/`n/a` means the telemetry was unavailable, which does **not** mean no throttling occurred._

## 4. Baseline and peak memory

| Job | Peak RSS (GB) | Peak extra system memory (GB) | Min available (GB) | Max system use (%) |
|---|---|---|---|---|
| performance@cpu:mlp | 0.478 | 0.0765 | 14.85 | 5.40 |
| neuroevolution@cpu | 0.453 | 0.0347 | 14.86 | n/a |

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
| neuroevolution@cpu | completed | 16 | 0 | 3.45 | 4.64 | 0.274 | 0.719 | 0.723 | [128r-32r-32s] lr=1.73e-02 |

## 7. Time per generation

**neuroevolution@cpu**

| Gen | Evaluated | Failed | Wall (s) | Ind/s | Mean fit time (s) | Best val R2 | Mean val R2 | CPU cores busy | System CPU % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 0 | 1.49 | 4.03 | 0.204 | 0.445 | 0.0973 | 1.12 | 29.10 |
| 1 | 5 | 0 | 1.38 | 3.62 | 0.224 | 0.705 | 0.339 | 1.06 | 27.10 |
| 2 | 5 | 0 | 1.42 | 3.53 | 0.228 | 0.719 | 0.623 | 1.07 | 27.30 |

## 8. Memory growth per generation / diagnostic

**neuroevolution@cpu**

| Gen | RSS before cleanup (GB) | RSS after (GB) | Extra system after (GB) | Peak extra (GB) | Available after (GB) | Swap since start (MB) | Cleanup (s) |
|---|---|---|---|---|---|---|---|
| 0 | 0.433 | 0.433 | 0.0235 | 0.0235 | 14.87 | 0 | 0.105 |
| 1 | 0.445 | 0.445 | 0.0292 | 0.029 | 14.87 | 0 | 0.113 |
| 2 | 0.453 | 0.453 | 0.0349 | 0.0347 | 14.86 | 0 | 0.116 |

Growth per generation (calculated): system 5.79 MB, RSS 10.56 MB. Rising RSS alone is not a leak: allocator caching also grows then plateaus.

**Memory-growth diagnostic (`memdiag`, each policy in its own process; verdicts are heuristic):**

| Job | Status | RSS verdict | RSS MB/cycle | System verdict | System MB/cycle | Python MB/cycle |
|---|---|---|---|---|---|---|
| memdiag@cpu:none | completed | continued_native_growth | 3.33 | continued_native_growth | 1.92 | 0.374 |
| memdiag@cpu:gc | completed | continued_native_growth | 3.38 | continued_native_growth | 3.10 | 0.399 |
| memdiag@cpu:clear | completed | continued_native_growth | 2.37 | continued_native_growth | 2.06 | 0.404 |

_`bounded`/`saturating` = consistent with allocator caching; `python_retention` = references still held; `continued_native_growth` = potential native leak, needs a longer run to confirm._

## 9. Incomplete or aborted tests

| Job | Status | Reason |
|---|---|---|
| performance@gpu:mlp | unsupported | no GPU is visible to TensorFlow (is tensorflow-metal installed and compatible?). Refusing to fall back to the CPU silently. |

## 10. Measurement limitations and notes

- Training time per individual includes per-model graph tracing/compilation (realistic for neuroevolution, where every individual is a new model).
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
