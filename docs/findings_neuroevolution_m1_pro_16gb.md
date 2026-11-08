# Neuroevolution CPU/GPU findings: MacBook Pro 14" (2021), Apple M1 Pro, 16 GB

**Provenance.** Numbers are copied from `tools/scaling_digest.py` output produced on the machine itself, in six runs on 2026-11-08:
four `--quick` runs (`...T070051Z`, `T131936Z`, `T132922Z`, `T134011Z`), one full run (`T135128Z`) and one targeted step-extension run
(`T141617Z`). Raw reports stay in `bench_results/` on that Mac (git-ignored). Nothing here is extrapolated to other hardware.

**Setup.** TensorFlow 2.15.1, tensorflow-metal 1.2.0, macOS 27.0.1, `--grappler none` (required: the default graph optimizer breaks the GPU),
`SimpleAdam`, strict GPU placement, batch 128, input dim 32, synthetic regression, activations relu/tanh, seed 1234, G=3 generations (quick runs G=2,
pilot P=4). Tiers: small (1-2 layers x 32-64), medium (2-4 x 128-256), large (4-6 x 512-1024; ~4M parameters).

## 1. Sequential evaluation (identical strategy: CPU sequential vs GPU sequential, heterogeneous populations)
GPU/CPU individuals per second (< 1 = CPU faster). "conf." = 3 independent repetitions, 95% CI.

| Tier | S=1 | S=5 | S=20 | S=50 | S=100 | S=200 |
|---|---|---|---|---|---|---|
| small (P=8) | 0.17 | 0.18 | 0.16 | 0.15 | - | - |
| medium (P=8) | 0.17 | 0.17 | 0.17 | 0.18 | - | - |
| large (P=8) | 0.19 | 0.23 | 0.37 | 0.59-0.63 (conf. **0.654 [0.610, 0.701]**) | 0.82 | 1.05 (conf. **1.04 [0.999, 1.08]**, inconclusive) |

* Large tier, S=50, other populations (conf.): P=4 **0.613 [0.575, 0.653]**, P=16 **0.574 [0.528, 0.625]**; P=32 0.568 (single run). Population size does not matter for
  sequential evaluation, as expected.
* Large tier, S=20, repeated confirmations across four quick runs: 0.383 [0.372, 0.393], 0.373 [0.346, 0.403] (P=8); 0.400 [0.395, 0.405], 0.394 [0.391, 0.398] (P=4).
  The ratio reproduces to within a few percent across sessions.
* **No confirmed GPU win in sequential evaluation up to S=200.** Parity (1.04, CI touching 1) is reached at ~200 steps per individual for ~4M-parameter networks at batch 128.
  The fitted cost model (GPU ~0.66-0.72 s per new model + 8.2-9.1 ms/step; CPU ~0.14-0.16 s + 10.0-11.8 ms/step) gave a break-even of 186-241 steps in six fits (one
  outlier at 587 in a quick run) and predicted this parity. Per-step the GPU is only ~1.1-1.34x faster at this size, so the sequential GPU advantage stays small even for many steps.
  Small and medium networks: GPU 5-7x slower at every tested step count (<= 50).

## 2. Batched evaluation of a homogeneous population (identical strategy: CPU-batched vs GPU-batched)
One shared architecture (5 x 1024) with different seeds/learning rates, trained as one stacked network.

| Setting | GPU-batched / CPU-batched (conf., 95% CI) | GPU-batched vs GPU-sequential | CPU-batched vs CPU-sequential |
|---|---|---|---|
| S=20, P=4 | 0.871 [0.841, 0.902]  (CPU faster) | 2.0-2.2x | 1.14-1.20x |
| S=20, P=8 | **1.25 [1.21, 1.29]** and 1.26 [1.23, 1.29] (GPU faster, two runs) | 3.29-3.31x (conf. [3.18, 3.40]) | 1.29-1.31x |
| S=50, P=8 | **1.73 [1.64, 1.82]** (GPU faster) | 2.71x (conf. [2.59, 2.84]) | 1.22x |

* **Confirmed crossover along population size** (S=20): between P=4 and P=8 (rough log-linear interpolation ~P=5). At P=8 the advantage grows with steps (1.25 -> 1.73).
* Batching a **heterogeneous** random population gives ~1.02-1.10x on both devices: only 20-29 % of individuals share a batchable (depth, activations) bucket.
* CPU multiprocessing (2 workers, 1 thread each, large tier) ran at 0.65-0.78x of sequential CPU: large networks benefit from the sequential run's multi-threaded matmul.
  4 workers were skipped for memory headroom.

## 3. Where the time goes (measured shares of wall time, sequential)
* GPU: per-model overhead dominates for few steps: first execution 40-52 %, teardown 14-20 %, construct+trace ~12-15 %, training 9-17 % (S <= 20). With more steps training
  grows (33 % at S<=50, 61 % up to S=200).
* CPU: small/medium nets spend 40-46 % in tracing and 15-17 % in construction; large nets spend 52-91 % in training (more with more steps).
* Teardown (dropping the model/optimiser/function/buffers) is ~0 % on CPU and 14-20 % on GPU (cause not isolated). Garbage collection per generation costs up to ~24 % of tiny CPU cells.

## 4. Memory
No swap growth anywhere. The same cell (GPU-batched, large, P=8, homogeneous) used 1.16 GB, 1.74 GB and 1.77 GB in three sessions, so peak additional memory is too
session-dependent to support a scaling law; an earlier "~0.2 GB per individual" estimate was withdrawn. P=32 batched cells and 4-worker multiprocessing did not fit the available
headroom (3.1 GB at best) and were skipped, not run.

## 5. What this does and does not support
* **Supported:** for sequential evaluation of freshly built networks (<= 50 steps, <= ~4M parameters, batch 128) the CPU is faster, confirmed (up to ~2.6x at S=20, ~1.5x at S=50 large).
  The GPU wins only when a homogeneous population is evaluated as one batch (1.25-1.73x vs the CPU's batched strategy, P >= 8), and approaches parity in sequential mode near ~200 steps.
* **Engine design hypotheses (not tested):** ~80 % of GPU time at low step counts is per-model overhead, so amortising it (batching, reusing traced functions, keeping models resident)
  is the lever; random topologies defeat batching (20-29 % coverage), grouping individuals by architecture is the obvious way to create batchable buckets.
* **Hardware decision:** which engine design you build decides whether GPU capability matters (batched homogeneous) or CPU speed does (heterogeneous sequential). This suite cannot predict
  other chips; run the identical commands on the candidate Mac with the same versions and compare with `tools/compare_reports.py`.
* **Not measured:** batch sizes other than 128 (the GPU's per-step efficiency is far below the earlier large-batch benchmark, so this is the most important open axis), populations above 8 for
  batching, medium/small tiers in batched mode, steps above 50 for other tiers, day-to-day variance beyond the repeats above.
