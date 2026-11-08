# Neuroevolution CPU/GPU findings: MacBook Pro 14" (2021), Apple M1 Pro, 16 GB

**Provenance.** Numbers are copied from `tools/scaling_digest.py` output produced on the machine itself, in runs on 2026-11-08:
four `--quick` runs (`...T070051Z`, `T131936Z`, `T132922Z`, `T134011Z`), one full run (`T135128Z`), one targeted step-extension run
(`T141617Z`), a batch-512 quick run (`T191017Z`), a batch-512 extended-steps run (`T192046Z`) a batch-1024 extended-steps run (`T194917Z`), a memory-capacity/concurrency run (`T210559Z`) and a memory-lifecycle run (`T214414Z`) with a concurrent `memwatch` recording. Raw reports stay in `bench_results/` on that Mac (git-ignored). Nothing here is extrapolated to other hardware.

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

## 8. Memory capacity and concurrency (run `...T210559Z`; `--scaling-design capacity`; large tier, S=20, G=2, batch 128, single runs, no CIs)
Run status: completed in 5.8 min of 20; planning headroom **4.3 GB** (the 80 % system-use ceiling with macOS and background apps using ~9.6 GB of 16 GB; limits not raised); no aborts, **0 MB swap growth anywhere**.
Skipped (never counted as measurements): GPU-batched P=32 (projected 3.25 GB from max(closed-form estimate, probe slope), x1.5 margin > 4.3 GB) and all four CPU 4-worker cells (estimated 3.1 GB x1.5 > 4.3 GB).

**Resident-model probe (MEASURED; models built and kept alive, 1..32; additional system memory):**
| | 1 | 4 | 8 | 16 | 32 | fit (CALCULATED) |
|---|---|---|---|---|---|---|
| GPU large (78.5 M params at 32) | 0.60 GB | 0.85 | 0.96 | 1.11 | 1.57 | 0.67 GB fixed + 28 MB/model (12.4 B/param) |
| CPU large | 0.11 | 0.11 | 0.36 | 0.62 | 0.96 | 0.07 GB fixed + 29 MB/model (12.6 B/param) |
| GPU medium (2.9 M params at 32) | 0.23 | 0.48 | 0.90 | 1.59 | 2.87 | ~85 MB/model (~990 B/param) |
| CPU medium | 0.00 | 0.00 | 0.13 | 0.13 | 0.39 | ~13 MB/model |
* Large-tier resident memory is **weights + two Adam moments = 12 B/parameter** (the 16 B/param theory additionally counts the transient gradient). 32 resident large models cost 1.6 GB on the GPU and 1.0 GB on the CPU.
* **Medium-tier GPU memory per resident model (~85 MB although the weights are ~0.4 MB):** reproduced and localised in section 9 (it arises when each model's training function is first executed, not from variables or tracing). The contradiction with the large tier (~28 MB per model) is still unexplained, so the large-tier projection should not be applied to small models.
* **Release:** after dropping 32 models and `gc`, the system got back 46 % (GPU large), 0 % (CPU large), and nothing (GPU medium: 2.9 GB still held). Process memory is not reliably returned to macOS; reuse inside the same process was not tested.

**Strategy grid (MEASURED):**
| P | GPU-seq | CPU-seq | GPU-batched | CPU-batched | CPU mp2 |
|---|---|---|---|---|---|
| 4 | 0.98 | 2.49 | 2.34 | 2.77 | 1.66 |
| 8 | 1.21 | 3.00 | 3.79 | 2.99 | 2.44 |
| 16 | 1.21 | 3.11 | 4.88 | 3.07 | 2.90 |
| 32 | 1.11 | 2.98 | skipped | 3.07 | 2.64 |
(individuals per second; GPU-batched/CPU-batched = 0.84, 1.27, 1.59 at P=4, 8, 16, the first two reproducing the earlier 0.87 and 1.25 within the earlier CIs.)
* **Sequential population size is not resident memory:** peak additional memory of the sequential strategies was 0.12-0.53 GB for every P (the P=4 value contains first-use runtime start-up), i.e. one model alive at a time whatever P is; throughput is flat in P once start-up is amortised. Batched keeps P models alive: CPU-batched peaked at 0.42 / 0.28 / 0.84 / 1.97 GB for P=4/8/16/32; GPU-batched 0.98 / 1.17 / 1.77 GB for P=4/8/16.
* **GPU-batched memory accumulated across cells of one process:** it held 0.98 / 1.17 / 1.13 GB after the P=4/8/16 cells (retained, system-wide) and minimum available memory fell 7.92 -> 6.75 -> 4.98 GB, i.e. about 3.9 GB were consumed cumulatively although every cell's own peak was <= 1.8 GB. Per-cell peaks therefore understate the footprint of a long-lived GPU process (an inference from these three cells). **Not reproduced in section 9:** in a controlled repeat (P=4 and 8, three identical rounds, with settling) the held memory plateaued at ~0.5 GB and later rounds added nothing, so this inference should be read as first-use growth plus system noise until P=16 is repeated under the same protocol.
* **Compute:** CPU-batched and CPU-sequential plateau at ~3.0 ind/s from P=8 (compute-bound, 1.00-1.03x of sequential); CPU mp2 (1 thread/worker) never beats sequential (0.67-0.93x); GPU-batched was **still rising at the largest tested P** (x1.29 for 8->16), i.e. not saturated at 16 resident models. P=32 on the GPU was not measured.
* **Framework overhead and construction:** GPU sequential spends 78-80 % of wall time in fixed per-model cost (construct+trace+first execution+teardown) vs 37-40 % on the CPU; building a model costs 0.6-0.9 s (GPU) vs 0.12-0.26 s (CPU). Sequential retained system memory was small (<= ~0.5 GB per cell; ~8 MB per evaluation on the GPU, ~2 MB on the CPU at P=32).

**What this does and does not say about memory capacity.** No configuration that ran needed more than 2 GB for a single cell (3.9 GB cumulative in one GPU-batched process); the workload as tested is far below 16 GB. The configurations that could not be run were skipped because of the headroom of this machine under its normal load (4.3 GB) with a 1.5x safety margin, not because they were measured to exceed 16 GB (the estimate for the largest skipped cell is ~3.3 GB). The calculated resident capacity (large tier, 12.4 B/param) is ~190 models per 6 GB of free memory; this is a linear extrapolation of the probe that excludes the unexplained small-model GPU overhead and is not a measurement.

## 9. Memory retention, process lifecycle and the medium-tier overhead (run `...T214414Z`; `--scaling-design lifecycle`; GPU-batched homogeneous large tier, S=20, G=2, batch 128)
**What was and was not done.** Planning headroom was only **2.8 GB** (other apps), so all three **P=16 cells were skipped** (estimate 2.1 GB x1.5 > 2.8 GB): the requested P=4/8/16 sequence was reproduced for **P=4 and P=8 only**, three identical rounds each. No aborts, swap growth 0 MB. Elapsed 4.0 min.
Values below are GB relative to the moment the TensorFlow runtime was ready (long-lived process) or relative to the cell start (fresh processes). System-wide available memory fluctuates by roughly +-0.2 GB on its own (e.g. "after gc" 0.08 then "settled" -0.18 in the first cell), so smaller differences are not interpreted.

**Long-lived process (MEASURED), memory held after settling (+4 s) after each cell:** round 0 P4 -0.18, P8 0.54; round 1 P4 0.50, P8 0.51; round 2 P4 0.31, P8 0.53. Held at the end of each round: 0.54 / 0.51 / 0.53 GB -> **stabilised**: the first round grows the process to ~0.5 GB, rounds 2 and 3 add nothing (+0.01 GB), i.e. the memory is reused by later cells. The OS-accounted footprint of the process behaved the same way: 0.41 GB at runtime-ready, 2.38 after the first P=8 cell, then 2.41, 2.41, 2.43, 2.44 (a plateau, +0.06 GB over four further cells). Process RSS (+0.3-0.6) and USS (+0.1) fluctuate without trend.
* **Peak vs held:** right after a cell the system held 0.4-1.0 GB, about half of it returned within seconds (after gc and settling). Throughput did not drift (P4 2.18/2.18/2.35, P8 3.17/3.20/3.39 ind/s).
* **Layers disagree and that is itself a finding:** after the first P=8 cell the OS footprint had grown by ~2.0 GB, process RSS by 0.3 GB, process USS by ~0.1 GB, and system-wide available memory by only ~0.5 GB. The OS `footprint` charges the process with graphics memory that is not mapped into its pages, so RSS and USS clearly understate what the GPU runtime holds. Why the footprint growth is not fully visible as lost available memory (purgeable/reclaimable or compressed graphics memory are candidates) is **unresolved**; no claim is made that 2.0 GB were physically unavailable to other apps.
* **No leak is indicated for P=4/8:** growth stopped, and the later fresh-process rows show the OS reclaims everything on exit. Allocator/pool caching that grows to the size of the largest cell and is then reused is the consistent reading; this was not shown for P=16.

**Fresh process per cell (MEASURED):** the same cells, one process each (~19 s wall per block including start-up and settling). Inside a process the cell held 0.87-1.03 GB (P4) and 1.49-1.72 GB (P8) right after execution and 0.43-0.49 / 0.84-0.89 after settling; process USS grew only 0.03-0.07 GB and RSS 0.4-0.5 GB, so most of it is outside the process's private pages. **After process exit the system returned to the pre-spawn level** (net held -0.04 to +0.10 GB in the four clean rows). The first two rows (net -2.05, -2.16 GB, and a P=8 "settled" of -1.56) are contaminated: available memory rose by ~2 GB across them, most plausibly because the long-lived process from the previous block was still being torn down (it had an OS footprint of 2.4 GB), which is itself consistent with that footprint being physical memory returned at exit; the controller now waits 2 s before taking its pre-spawn sample. Throughput inside a cell equals the long-lived process (P4 2.20-2.30, P8 3.05-3.58 ind/s); the cost of fresh processes is the ~13 s of start-up per cell, not slower execution.
**Comparison:** in a fresh process each cell needs ~0.4-0.9 GB that is returned on exit; in a long-lived process the first cell grows the process once and later cells reuse it (net ~0 per cell). Neither shows accumulation over three rounds at P<=8; fresh processes bound the footprint by construction.

**Medium-tier GPU overhead (MEASURED, fresh process each, 1..32 resident models):**
| variant | models reached | additional at max | system GB/model | process USS GB/model | OS footprint GB/model |
|---|---|---|---|---|---|
| build + trace + execute, heterogeneous | 32 | 1.97 GB | 0.057 | 0.092 | 0.094 |
| build + trace + execute, one shared architecture | 16* | 1.06 GB | 0.052 | 0.091 | 0.092 |
| trace only (never executed) | 32 | 0.19 GB | 0.001 | 0.001 | 0.002 |
| variables + optimiser state only | 32 | 0.17 GB | 0.000 | 0.000 | 0.001 |
(*the 32-model rung was refused by the 90 % projection guard at 2.8 GB headroom.)
* **The ~90 MB per model appears only when the model's training function is executed.** Variables, optimiser state and graph tracing cost essentially nothing (intercept ~0.13-0.19 GB). It is **not** per-shape specialisation: identical architectures cost the same (0.091 vs 0.092 GB/model). It is inside the process (USS and footprint agree at ~0.092 GB/model; the system-wide slope of 0.052-0.057 is smaller because the available-memory metric is noisier and partly offset by reclaim).
* **Consequence for a design:** each executed `tf.function` instance that stays alive on the GPU costs ~90 MB in the medium tier regardless of weight size; 32 resident medium models = ~2-3 GB, 100 = ~9 GB. The batched strategy (one stacked function for P models) does not pay this per model, and sequential evaluation releases each function; so the exposure is specific to designs that keep many separately executed functions alive. Whether sharing one traced function across same-architecture models removes it, or why the large tier shows only ~28 MB per executed model, was **not tested**.

**Concurrent `memwatch` recording (scenario "BENCH", 3.5 min, 43 samples):** memory pressure level normal in 100 % of samples, swap used 0 GB, compressed memory max 2.0 GB, available memory min 6.5 GB (median 8.4), demand (wired+active+compressed) 7.4 GB at the first minute, max 8.7 GB, CPU p95 42 %, scheduler lag p95 5.3 ms, spawn latency p95 24 ms, python process group peak 1.16 GB RSS.
**Caveat:** by the file timestamps the monitor was started about two minutes after the benchmark (assuming the Mac's local time is UTC+2), so its "starting level" already contained benchmark activity and the reported increase (+1.2 GB at 5 s resolution) is not an idle baseline; the benchmark's own 0.5 s sampler saw 1.6-1.8 GB additional at P=8. There is no scenario-A recording yet, so the responsiveness numbers have nothing to be compared with.

**Memory budget so far (what can and cannot be separated):**
1. *Memory the ML workload requires:* weights + two Adam moments = 12 B/parameter per resident model (large tier); sequential cells peak at 0.2-0.5 GB; GPU-batched cells 1.0-1.8 GB (P=4-16 in the capacity run; P=8 1.6-1.8 GB in this run), plus a fixed GPU-runtime component of 0.4-0.7 GB.
2. *Memory retained by TensorFlow/Metal:* ~0.5 GB system-held (2.0 GB OS footprint, partly unexplained) in a long-lived GPU process after the first P=8 cell, stable thereafter; fully returned on process exit. P=16 untested.
3. *Memory used by normal applications:* ~7.4-8.7 GB demand in this session (macOS plus the usual login apps), **not yet a clean baseline**.
4. *Extra headroom for concurrent workloads (ML + Lightroom + Photoshop):* **not measured** (scenario C missing).

## 10. What this does and does not support
* **Supported:** at batch 128 the CPU is faster for sequential evaluation (confirmed up to ~2.6x at S=20, ~1.5x at S=50 for the large tier) and the sequential GPU only reaches parity near ~200 steps;
  the GPU wins at batch 128 only when a homogeneous population is evaluated as one batch (1.25-1.73x vs the CPU's batched strategy, P >= 8).
  **At batch 512 the large tier flips: GPU sequential is confirmed faster from ~37 steps (1.22x at S=50, 2.05x at S=200)** and the batched crossover moves below P=4. At batch 1024 the GPU is confirmed faster at S=50 (1.67x) and S=200 (2.46x) and at parity at S=20 (0.96x); estimated break-even ~22 steps.
  Whether the GPU wins is therefore decided by compute per step (network size x batch) and steps per individual, with per-model overhead (~0.6 s on the GPU) as the cost to amortise.
* **Engine design:** the measured rule of thumb for this machine is: GPU when (large networks AND batch >= 512 AND >= ~40 steps/individual) or a homogeneous population can be batched; CPU for small/medium networks, short training, or heterogeneous topologies at batch <= 128.
  Hypotheses (not tested): ~80 % of GPU time at low step counts is per-model overhead, so amortising it (batching, reusing traced functions, keeping models resident)
  is the lever; random topologies defeat batching (20-29 % coverage), grouping individuals by architecture is the obvious way to create batchable buckets.
* **Hardware decision (batch 512/1024 and capacity data):** (a) Memory: every sequential configuration peaked at ~0.6 GB additional on a 16 GB machine with zero swap; the largest measured peak anywhere was ~1.8 GB (batched P=8). Nothing measured here shows
  memory capacity limiting the workload, so these results give **no evidence that 48, 64 or 96 GB would speed up this workload**; larger populations held concurrently (not measured: P=32 batched and 4-worker runs were skipped for headroom) could change that and should be measured before buying for memory.
  (b) Processor: GPU per-step time did not respond to a 2x batch increase, i.e. the sequential GPU path is not compute-limited at these sizes, so extra GPU cores alone are not supported as a benefit for this path by the data; whether a Max-class chip helps depends on engine design
  (large batched/compute-heavy work vs per-step overhead, which depends on single-thread CPU speed and the framework), which is untested. These are readings of M1 Pro measurements only; no newer chip was measured and none is inferred.
* **What can be concluded about 48 / 64 / 96 GB (memory runs so far):** (a) nothing measured shows the ML workload needing more than a few GB: every configuration that ran fit in a headroom of 2.8-4.3 GB, retained memory plateaued and was returned on process exit. So there is **no workload-based evidence for 48, 64 or 96 GB**, and also none against it for designs not yet tested. (b) The one measured memory multiplier is ~90 MB per separately executed training function held alive on the GPU (medium tier); a design that keeps hundreds of such models resident could reach tens of GB, a design that batches or releases them does not. That is a design choice, not a hardware fact. (c) Whether 16 GB is enough *alongside* Lightroom Classic and Photoshop, and how much headroom the combination needs, is **not yet measured**; it requires the scenario A/B/C recordings. (d) Nothing here concerns newer chips or a Pro/Max choice beyond the earlier statements.
* **Hardware decision:** which engine design you build decides whether GPU capability matters (batched homogeneous) or CPU speed does (heterogeneous sequential). This suite cannot predict
  other chips; run the identical commands on the candidate Mac with the same versions and compare with `tools/compare_reports.py`.
* **Not measured (capacity run):** GPU-batched P=32 and CPU 4-worker multiprocessing (skipped for headroom); repeats/CIs for any capacity cell; the small-model GPU memory overhead (unexplained); whether process memory is reused after release; concurrent CPU+GPU use of memory; tiers other than large in the strategy grid.
* **Not measured:** batch sizes other than 128, 512 and 1024 (larger than 1024 untested; at 1024 only the large tier, P=4, sequential, and no S below 20), populations above 8 for
  batching, medium/small tiers in batched mode, steps above 50 for other tiers, day-to-day variance beyond the repeats above.
