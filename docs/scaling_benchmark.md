# Neuroevolution CPU/GPU scaling and crossover benchmark (`--tests scaling`)

**Research question.** Under which combinations of network complexity, population size, training workload and execution
strategy does the Metal GPU outperform the CPU for neuroevolution on Apple Silicon? *A crossover is not assumed to exist.*
The benchmark is designed so that "no crossover in the tested region" is an acceptable, properly evidenced outcome.

## Quick start (M1 Pro, 16 GB) and full run

```bash
python -m unittest discover -s tests -t .                                         # 1. unit tests
python tf_memory_stress_test.py --tests scaling --quick --grappler none --dry-run # 2. estimates, trains nothing
python tf_memory_stress_test.py --tests scaling --quick --grappler none --max-gb 6   # 3. quick run (~5-10 min, hard cap 10 min)
python tf_memory_stress_test.py --tests scaling --grappler none --max-gb 6           # 4. full run (~15-30 min, hard cap 30 min)
python tools/recommend_device.py bench_results/RUN/report.json --params 5e6 --layers 6 --steps 50 --population 16   # advice only
```
`--grappler none` is required on setups where the default graph optimizer breaks the GPU (see README *Known issue*); the numerical
gate at the start of every block refuses to report timings otherwise.

## Estimated size, runtime and resource exposure (computed by `--dry-run`; PRIORS, +-2x)

| | Quick | Full |
|---|---|---|
| Exhaustive grid (tiers x populations x steps), NOT run | 24 workload points | 48 workload points |
| Pilot (paired CPU/GPU sequential, 1 repetition) | 12 cells | 24 cells |
| Expansion (upper bound; cut by budget) | <= 17 cells | <= 46 cells |
| Confirmation (upper bound; 3 independent repetitions) | <= 21 cells | <= 33 cells |
| **Cell-runs in total (upper bound)** | **<= 50** | **<= 103** |
| Worker processes (each block = one fresh process) | <= 22 | <= 22 |
| Typical runtime / plausible range (measured quick run on an M1 Pro: ~5 min for 39 cells) | ~7 min / 4-15 min | ~26 min / 13-51 min |
| Hard caps | 2.5 min per configuration, 10 min total | 5 min per configuration, 30 min total |
| Worst-case single configuration, memory (estimated) | ~1.6 GB | ~3.6 GB (large, P=32, batched) |

The runtime priors come from the author's M1 Pro baseline (GPU ~1.4-1.8 s per freshly built model, CPU ~0.15 s) and are replaced by
measured costs after the pilot. When the estimate exceeds the budget the controller **trims**, it never overruns: lower-priority
expansion cells are skipped and every skip is recorded. Resource exposure is bounded by the safety limits (6 GB extra memory,
2 GB available floor, 80 % system use, 512 MB swap), by planning against the *current* memory headroom (the smallest of the three
memory limits), and by the per-block watchdog and external supervisor described in the main README.

## Dimensions

| Tier | Hidden layers | Units per layer | Canonical architecture (homogeneous populations) |
|---|---|---|---|
| small | 1-2 | 32, 48, 64 | 64-64 |
| medium | 2-4 | 128, 192, 256 | 256-256-256 |
| large | 4-6 | 512, 768, 1024 | 5 x 1024 |

Populations 4, 8, 16, 32; training steps per individual 1, 5, 20, 50 (exactly that many optimiser steps, batch 128, `SimpleAdam`);
G = 3 generations (2 in quick mode) of tournament selection + elitism (25 %) + crossover + mutation, so the genetic operators are
part of the measured work. Fitness = validation R2 on a reproducible synthetic regression task.

Strategies: **cpu-seq**, **gpu-seq**, **cpu-mpW** (spawned worker processes, W in {2, 4}, 1 thread each), **cpu-batched**,
**gpu-batched**. GPU multiprocessing is intentionally unsupported (`--scaling-allow-gpu-multiprocess` is rejected).

### Which comparisons are like-for-like?
* **Identical strategy** (same workload, same execution method, different device): cpu-seq vs gpu-seq, cpu-batched vs gpu-batched.
  *These are the only comparisons that define a device crossover.*
* **Different strategy** (labelled "cross-strategy" everywhere): batched/multiprocess vs sequential on the same device, and the
  "best available strategy on each device" table. They answer "what should an engine do", not "which silicon is faster".
* Paired runs share a *workload signature* (genomes, data spec, seed, budget) that contains nothing about the strategy; this is unit-tested.

## Adaptive design (no Cartesian product)
1. **Pilot** - paired CPU/GPU sequential runs over tier x steps at one population size (8; 4 in quick mode). Exploration, 1 repetition.
2. **Candidates** - the best GPU/CPU ratio, both sides of any sign change of (ratio-1) along steps or model size, then points near
   parity. If the GPU is hopeless everywhere the best point is still confirmed so a negative result is evidenced, not assumed.
3. **Expansion** - around the candidates, in order of information value: nearest other population size, batched (heterogeneous and
   homogeneous populations, with sequential baselines on the same workload), CPU multiprocessing (heterogeneous AND homogeneous
   populations, so the best CPU strategy is known wherever batching is tested), remaining sizes, larger populations,
   and (opt-in `--scaling-extend-steps`) step counts beyond the grid.
4. **Pruning** - a configuration is skipped when its predicted time exceeds the per-configuration limit, its estimated memory x1.5
   exceeds the headroom, an equal-or-cheaper configuration of the same strategy/tier already aborted, or the phase budget is spent.
5. **Confirmation** - the candidates (and, when batching is the best strategy gain, the batched strategy on BOTH devices at that point and at the nearest
   other population size, so a crossover along population size can be confirmed) are repeated `--scaling-confirm-reps` times (default 3), each repetition in **fresh processes**
   with the CPU/GPU order alternated, so process state and slow drift are not shared. Only these repetitions feed confidence intervals.

## Timing decomposition (per cell, from host read-back barriers; no per-batch callbacks)
`construct` (model/weights/optimiser/batch upload) - `trace` (tf.function tracing) - `first_exec` (graph optimisation, kernel/shader
compilation, dispatch warm-up **and training step 1**) - `train` (steps 2..S) - `fitness` (validation forward + R2) - `ga` (selection,
crossover, mutation) - `cleanup` (`gc.collect`) - `unattributed` (wall minus the above). Multiprocess cells report *worker-seconds* and
a parallel efficiency instead of a wall attribution. A discarded **warm-up cell** per process absorbs runtime/shader-cache cold start;
its cost is recorded as `cold_start_s`. Reported per configuration: individuals/s, training steps/s, wall time per generation,
median and CV across repetitions, GPU/CPU ratio with CI, peak additional memory, swap growth, memory retained after cleanup.

## Batching (what is and is not claimed)
Arbitrary heterogeneous networks are **not** batched. Individuals are *bucketed* by (depth, per-layer activations); buckets of >= 2 are
trained as one population-stacked network with batched matrix multiplication. Narrower members are **zero-padded to the widest member and
masked**, which is mathematically identical to separate training (padded weights receive zero input, zero output and zero gradient, and
stay exactly 0). This is verified against sequential training in the unit tests and again on every device at the start of a batched block
(`selfcheck_batched`). Singletons run sequentially inside the same cell. Reported: share of individuals that were batchable
(**coverage**) and the padding overhead (padded / real parameters). Homogeneous populations (one shared architecture, evolved numerically)
are the best case and are measured separately.

## Multiprocessing safety
`spawn` start method (never fork an initialised TensorFlow runtime); the process owning the pool never imports TensorFlow; each worker hides
the GPU and is limited to 1 intra-op + 1 inter-op thread; worker count <= min(requested, `--scaling-max-workers`, memory-derived cap of
0.75 GB per worker within half the allowance); summed worker RSS is polled and the whole pool is terminated above 80 % of the allowance
or on any limit violation, time-out or Ctrl-C.

## Statistics
Throughput ratios are analysed in log space: `log r = mean(log gpu) - mean(log cpu)` with a Welch t-interval (95 %). Verdicts:
`gpu_faster` / `cpu_faster` only when the interval **excludes parity**, `inconclusive` when it straddles 1, `unconfirmed` with fewer
than 2 repetitions on either side. A **crossover is declared only between two adjacent confirmed points that change from cpu_faster
to gpu_faster** (along training steps and model size for sequential evaluation, along population size for the batched strategy); an exploration-only sign change is reported as an *unconfirmed candidate*. With `n = 3` intervals are wide - that
is reported, not hidden. A two-parameter cost model per device and tier (`fixed seconds per new model` + `seconds per step`, fitted
to the measured components) gives an **estimated** break-even `S* = (fixed_gpu - fixed_cpu) / (step_cpu - step_gpu)`; above the tested
range this is an extrapolation and is labelled as one.

## Execution-policy advisor (recommendation mode only)
`tfbench/scaling/scheduler.py` and `tools/recommend_device.py` predict population time per strategy from the calibrated cost model
(inputs: parameters, layers, steps, batch, population, available memory) and print a recommendation with confidence and warnings. It never
changes a device during a benchmark. Against overfitting one machine/session it is validated by **leave-one-tier-out** error, flags
extrapolation beyond 2x the calibrated range, records the chip it was calibrated on and warns on mismatch, and caps confidence at *low*
when its held-out error exceeds 30 %. Strategy gains (multiprocessing, batching) enter only as measured multipliers.

## Failure handling
| Status | Meaning | Counted as a measurement? |
|---|---|---|
| completed | ran inside all limits | yes |
| aborted | per-configuration time limit or a resource limit fired | **no** (listed, prunes costlier cells) |
| failed | unexpected error or failed numerical/placement validation | **no** |
| unsupported | device missing / unsupported GPU kernel under strict placement | **no** (no CPU fallback, ever) |
| skipped | budget, memory estimate or pruning | **no** (listed with the reason) |
| soft placement GPU cells | GPU run with CPU fallback allowed (`--placement soft`) | shown, but **excluded from all GPU conclusions** |

A missing GPU disables every GPU strategy after one spawn; a failed numerical gate disables every strategy on that device; a
strategy-specific failure (e.g. one unsupported kernel) only that strategy. Unfinished cells of a crashed block are re-queued once in a fresh process.

## Outputs
`report.json` (job `scaling`: cells, blocks, skipped, analysis, scheduler model), `measurements.csv`, `summary.md` (section 11), and
`scaling/` with `scaling_summary.md`, `cells.csv`, `crossover.json` and SVG charts (`ratio_vs_steps`, `ratio_vs_model_size`, `heatmap_ratio`,
`time_components`). `tools/compare_reports.py` lines up scaling ratios across runs/machines.

## Limitations
One machine and session (day-to-day throughput variance of up to ~6 % is NOT in the CIs); small n per confirmed point; cold vs warm Metal
shader cache changes first-execution cost; synthetic regression and small dense networks only; per-model Keras/TF overhead and the
tensorflow-metal plugin cannot be separated from the M1 Pro GPU hardware; GPU reductions are not bit-reproducible; a native call blocked
inside Metal can only be stopped by the supervisor; break-even steps beyond the tested range are extrapolations; **no performance is
inferred for any chip that was not benchmarked** - compare machines only by running this suite on them.
