# Neuroevolution CPU/GPU findings: MacBook Pro 14" (2021), Apple M1 Pro, 16 GB

**Provenance.** Numbers are copied from `tools/scaling_digest.py` output produced on the machine itself, in runs on 2026-11-08:
four `--quick` runs (`...T070051Z`, `T131936Z`, `T132922Z`, `T134011Z`), one full run (`T135128Z`), one targeted step-extension run
(`T141617Z`), a batch-512 quick run (`T191017Z`), a batch-512 extended-steps run (`T192046Z`) and a batch-1024 extended-steps run (`T194917Z`). Raw reports stay in `bench_results/` on that Mac (git-ignored). Nothing here is extrapolated to other hardware.

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

## 5. Batch size (one quick run at `--scaling-batch 512`, run `...T191017Z`; G=2, pilot P=4; compare with the batch-128 quick runs above)
| Measurement (large tier, S=20, P=4) | Batch 128 | Batch 512 |
|---|---|---|
| Sequential GPU/CPU (conf.) | 0.394 [0.391, 0.398], 0.400 [0.395, 0.405] | **0.666 [0.646, 0.687]** |
| Sequential CPU / GPU individuals per second | 2.47 / 0.99 | 1.47 / 0.98 |
| Batched homogeneous GPU/CPU (conf.) | 0.871 [0.841, 0.902] (CPU faster) | **1.70 [1.65, 1.75]** (GPU faster) |
| GPU-batched vs GPU-sequential (conf.) | 2.0-2.2x | 1.91x [1.86, 1.95] |
| CPU-batched vs CPU-sequential | 1.14-1.20x | 0.97x |
| Fitted per-step cost, GPU / CPU (estimated) | 8.2-9.1 ms / 10.0-11.8 ms | 9.6 ms / 25.9 ms |

* GPU throughput did not change when the batch quadrupled (per-step cost 8.7 -> 9.6 ms) while the CPU's per-step cost grew 2.2x: at these batch sizes the GPU is **overhead-bound**, not compute-bound.
* Larger batches therefore help the GPU relative to the CPU: the sequential per-step advantage rose from ~1.3x to ~2.7x, and the batched crossover along population size moved below P=4.
* The fitted model now gives a sequential break-even of ~39 steps for the large tier at batch 512 (ESTIMATED; the run only reached S=20, so this is an extrapolation to be tested).
* Small and medium tiers are unchanged (GPU/CPU 0.16-0.21): compute per step is negligible there. Heterogeneous batching stays ~1.0x (28.6 % batchable).
* Caveats: one quick run; CPU multiprocessing was skipped twice for memory headroom; batch sizes above 512 were not tested.

## 6. Sequential evaluation at batch 512 with extended steps (run `...T192046Z`; large tier, P=4, heterogeneous, G=3)
GPU/CPU individuals per second, 3 independent repetitions each (95% CI):

| Steps | CPU ind/s | GPU ind/s | GPU/CPU | Verdict |
|---|---|---|---|---|
| 20 | 1.43 | 0.97 | 0.673 [0.659, 0.687] | CPU faster |
| 50 | 0.631 | 0.762 | **1.22 [1.17, 1.27]** | **GPU faster** |
| 100 | 0.375 | 0.541 | 1.44 (single run) | unconfirmed |
| 200 | 0.176 | 0.363 | **2.05 [2.01, 2.09]** | **GPU faster** |

* **Confirmed sequential crossover along training steps between S=20 and S=50 (interpolated ~S=37)** at batch 512.
* The cost model fitted on the earlier batch-512 quick run predicted break-even ~39 steps and ratios of ~1.15 / 1.6 / 2.0 at S=50 / 100 / 200; measured 1.22 / 1.44 / 2.05 (within ~10 %).
  The new fit (GPU ~0.79 s per model + 10.2 ms/step; CPU ~0.16 s + 27.4 ms/step) gives a break-even of 37 steps. This validates the model for THIS tier only.
* Time split (S up to 200): GPU first execution 28 %, training 52 %, teardown 10 %; CPU training 92 %. The per-step advantage (27.4 / 10.2 ms) is ~2.7x.
* Small and medium tiers at batch 512 (quick run, S <= 20): GPU/CPU 0.16-0.21; the fitted GPU per-step cost is not lower than the CPU's there, so the model predicts no crossover at any step count
  (a model reading, not a confirmed measurement).
* Caveats: one tier, one population size, one batch size; CPU multiprocessing not measured at batch 512; batches above 512 untested.

## 7. Sequential evaluation at batch 1024 vs batch 512 (run `...T194917Z` vs `...T192046Z`; identical flags except `--scaling-batch`; large tier, P=4, heterogeneous, G=3)
Run status: completed in 11.9 min of the 20 min budget, 26 cells completed, 0 not completed, 0 skipped (planning headroom was 2.9 GB, so configurations estimated above 1.9 GB would have been skipped; none were).
GPU/CPU individuals per second, 3 independent repetitions per confirmed point (95% CI):

| Steps | CPU ind/s 512 -> 1024 | GPU ind/s 512 -> 1024 | GPU/CPU at 512 | GPU/CPU at 1024 | Verdict at 1024 |
|---|---|---|---|---|---|
| 20 | 1.43 -> 1.04 | 0.97 -> 0.98 | 0.673 [0.659, 0.687] | 0.964 [0.875, 1.06] | **inconclusive (CI spans parity)** |
| 50 | 0.631 -> 0.462 | 0.762 -> 0.758 | 1.22 [1.17, 1.27] | **1.67 [1.50, 1.85]** | GPU faster |
| 100 | 0.375 -> 0.259 | 0.541 -> 0.558 | 1.44 (single run) | 2.16 (single run) | unconfirmed |
| 200 | 0.176 -> 0.147 | 0.363 -> 0.358 | 2.05 [2.01, 2.09] | **2.46 [2.33, 2.60]** | GPU faster |

* **Measured:** GPU throughput is unchanged between batch 512 and 1024 (all four step counts within ~3 %), while CPU throughput falls 17-27 %. The GPU/CPU ratio rises by ~1.43x (S=20), ~1.37x (S=50) and ~1.20x (S=200).
  The advantage still grows with batch size, but by less than from 128 to 512 (the CPU's per-step cost grows 1.43x for a 2x larger batch, versus ~2.2x for 4x earlier).
* **Crossover (batch 1024):** the tool reports `candidate_unconfirmed`, because S=20 is a parity result (0.964, CI 0.875-1.06), so no confirmed sign change exists between two tested points. What is measured: GPU confirmed faster at S=50 and S=200; S=20 is indistinguishable from parity.
  ESTIMATED by log-linear interpolation of the two confirmed ratios (0.964 at S=20, 1.67 at S=50): ~S=22, matching the fitted cost-model break-even of 22 steps. Using the CI end points of the S=20 ratio gives a plausible range of "below S=20" to ~S=28.
  Both are interpolations, not measurements; no step count below 20 was tested at this batch size. At batch 512 the confirmed crossover was between S=20 and S=50 (~37), so it moved lower, to about 20-28 steps.
* Cost model (ESTIMATED, fitted to measurements): batch 1024 GPU 0.79 s/model + 10.35 ms/step, CPU 0.16 s/model + 39.2 ms/step; batch 512 GPU 0.79 s + 10.20 ms, CPU 0.16 s + 27.4 ms. The per-step advantage is ~3.8x (was ~2.7x). The fixed per-model costs came out the same in both fits.
* **Memory (measured):** peak additional 0.64 GB (CPU) / 0.63 GB (GPU), retained 0.43 / 0.49 GB, swap growth 0 MB (batch 512: 0.58 / 0.62 GB, retained 0.39 / 0.50 GB). Doubling the batch did not change memory use measurably (differences are inside the session-to-session noise documented in section 4).
* **Time split:** CPU 94 % training; GPU first execution 27 %, training 53 %, teardown 10 %, construct+trace 9 % - essentially identical to batch 512.
* **Bottleneck reading:** the GPU's per-step cost is flat from batch 128 to 1024 (about 8-10 ms), so for these networks it is limited by per-step framework/dispatch overhead (TensorFlow-Metal op launch, optimizer, synchronisation) rather than GPU arithmetic or memory. The CPU is compute-bound (per-step cost grows with batch).
  Not memory-bound: peak extra memory is ~0.6 GB. This is an inference from timings, not a profiler result; the per-step GPU cost was not decomposed further.
* **Not done / limitations:** one tier (large), one population size (4), heterogeneous sequential only; CPU multiprocessing and batched strategies were not run; S=100 is a single exploration run; batches above 1024 and S<20 untested; no other machine measured; the break-even is below or near the shortest step count tested (20).

## 8. What this does and does not support
* **Supported:** at batch 128 the CPU is faster for sequential evaluation (confirmed up to ~2.6x at S=20, ~1.5x at S=50 for the large tier) and the sequential GPU only reaches parity near ~200 steps;
  the GPU wins at batch 128 only when a homogeneous population is evaluated as one batch (1.25-1.73x vs the CPU's batched strategy, P >= 8).
  **At batch 512 the large tier flips: GPU sequential is confirmed faster from ~37 steps (1.22x at S=50, 2.05x at S=200)** and the batched crossover moves below P=4. At batch 1024 the GPU is confirmed faster at S=50 (1.67x) and S=200 (2.46x) and at parity at S=20 (0.96x); estimated break-even ~22 steps.
  Whether the GPU wins is therefore decided by compute per step (network size x batch) and steps per individual, with per-model overhead (~0.6 s on the GPU) as the cost to amortise.
* **Engine design:** the measured rule of thumb for this machine is: GPU when (large networks AND batch >= 512 AND >= ~40 steps/individual) or a homogeneous population can be batched; CPU for small/medium networks, short training, or heterogeneous topologies at batch <= 128.
  Hypotheses (not tested): ~80 % of GPU time at low step counts is per-model overhead, so amortising it (batching, reusing traced functions, keeping models resident)
  is the lever; random topologies defeat batching (20-29 % coverage), grouping individuals by architecture is the obvious way to create batchable buckets.
* **Hardware decision (what the 512/1024 data say):** (a) Memory: every sequential configuration peaked at ~0.6 GB additional on a 16 GB machine with zero swap; the largest measured peak anywhere was ~1.8 GB (batched P=8). Nothing measured here shows
  memory capacity limiting the workload, so these results give **no evidence that 48, 64 or 96 GB would speed up this workload**; larger populations held concurrently (not measured: P=32 batched and 4-worker runs were skipped for headroom) could change that and should be measured before buying for memory.
  (b) Processor: GPU per-step time did not respond to a 2x batch increase, i.e. the sequential GPU path is not compute-limited at these sizes, so extra GPU cores alone are not supported as a benefit for this path by the data; whether a Max-class chip helps depends on engine design
  (large batched/compute-heavy work vs per-step overhead, which depends on single-thread CPU speed and the framework), which is untested. These are readings of M1 Pro measurements only; no newer chip was measured and none is inferred.
* **Hardware decision:** which engine design you build decides whether GPU capability matters (batched homogeneous) or CPU speed does (heterogeneous sequential). This suite cannot predict
  other chips; run the identical commands on the candidate Mac with the same versions and compare with `tools/compare_reports.py`.
* **Not measured:** batch sizes other than 128, 512 and 1024 (larger than 1024 untested; at 1024 only the large tier, P=4, sequential, and no S below 20), populations above 8 for
  batching, medium/small tiers in batched mode, steps above 50 for other tiers, day-to-day variance beyond the repeats above.
