> **EXAMPLE ONLY - not a benchmark of your Mac.** Generated in a Linux x86-64 cloud sandbox with CPU-only
> TensorFlow 2.15.1 and `--quick` settings, to show the report structure. The GPU job is listed as *unsupported*
> on purpose: that is how the suite reports a missing GPU (it never falls back to the CPU silently).
> Do not read any number below as representative of Apple Silicon.

# TensorFlow benchmark report

*Run `example_report` - started 2026-11-06T23:05:57+00:00 - finished 2026-11-06T23:06:39+00:00 - suite v2.0.0*

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
| neuroevolution@cpu | completed | strict | False | graph (Keras model.fit) | True | /job:localhost/replica:0/task:0/device:CPU:0 | {'CPU': 4} | tfbench.SimpleAdam (Keras-Adam update rule, float32 state) |

## 3. CPU and GPU performance (steady state)

| Workload | Device | Status | Steps/s | Samples/s | ms/step | Median run (s) | CV | Min run (s) | Max run (s) |
|---|---|---|---|---|---|---|---|---|---|
| mlp | cpu | completed | 67.80 | 8,679 | 14.75 | 0.442 | 0.104 | 0.409 | 0.502 |
| mlp | gpu | unsupported | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

_Steps/s, samples/s, ms/step, median and CV are **calculated** from the measured per-repeat times._

**CPU vs GPU (calculated):**

_no data_

**Phase breakdown (measured, seconds) - never merged with steady-state time:**

| Job | Data gen | Data transfer | Model build | Optimizer init | Trace | First execution | Warm-up total | Compile overhead (estimated) |
|---|---|---|---|---|---|---|---|---|
| performance@cpu:mlp | 0.0361 | 0.0612 | 0.282 | 0.0504 | 0.28 | 0.271 | 0.0783 | 0.256 |
| performance@gpu:mlp | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Thermal / stability indicators:**

| Job | pmset throttling indicated | Drift last-vs-first third (x) |
|---|---|---|
| performance@cpu:mlp | unavailable | 0.881 |
| performance@gpu:mlp | unavailable | n/a |

_`None`/`n/a` means the telemetry was unavailable, which does **not** mean no throttling occurred._

## 4. Baseline and peak memory

| Job | Peak RSS (GB) | Peak extra system memory (GB) | Min available (GB) | Max system use (%) |
|---|---|---|---|---|
| performance@cpu:mlp | 0.476 | 0.124 | 14.86 | 5.40 |
| neuroevolution@cpu | 0.408 | 0.0343 | 14.94 | n/a |

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
| neuroevolution@cpu | completed | 16 | 0 | 3.18 | 5.03 | 0.188 | 0.717 | 0.696 | [128r-32r-32s] lr=1.73e-02 |

## 7. Time per generation

**neuroevolution@cpu**

| Gen | Evaluated | Failed | Wall (s) | Ind/s | Mean fit time (s) | Best val R2 | Mean val R2 | CPU cores busy | System CPU % |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 6 | 0 | 1.54 | 3.90 | 0.201 | 0.529 | 0.113 | 1.03 | 30.50 |
| 1 | 5 | 0 | 1.67 | 2.99 | 0.254 | 0.589 | 0.23 | 1.05 | 28.60 |
| 2 | 5 | 0 | 1.48 | 3.38 | 0.231 | 0.717 | 0.572 | 1.06 | 28.20 |

## 8. Memory growth per generation / diagnostic

**neuroevolution@cpu**

| Gen | RSS before cleanup (GB) | RSS after (GB) | Extra system after (GB) | Peak extra (GB) | Available after (GB) | Swap since start (MB) | Cleanup (s) |
|---|---|---|---|---|---|---|---|
| 0 | 0.394 | 0.394 | 0.02 | 0.021 | 14.95 | 0 | 0.105 |
| 1 | 0.403 | 0.403 | 0.0254 | 0.0225 | 14.95 | 0 | 0.112 |
| 2 | 0.409 | 0.409 | 0.0343 | 0.0285 | 14.94 | 0 | 0.112 |

Growth per generation (calculated): system 7.32 MB, RSS 7.84 MB. Rising RSS alone is not a leak: allocator caching also grows then plateaus.

**Memory-growth diagnostic (`memdiag`, each policy in its own process; verdicts are heuristic):**

| Job | Status | RSS verdict | RSS MB/cycle | System verdict | System MB/cycle | Python MB/cycle |
|---|---|---|---|---|---|---|
| memdiag@cpu:none | completed | bounded | 0.517 | bounded | 0.0481 | 0.098 |
| memdiag@cpu:gc | completed | bounded | 0.395 | bounded | -0.0562 | 0.0189 |
| memdiag@cpu:clear | completed | bounded | 0.0067 | bounded | -0.0615 | 0.019 |

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
