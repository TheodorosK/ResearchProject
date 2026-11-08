# Memory-capacity and concurrency benchmark (`--scaling-design capacity`)

Purpose: give evidence for the unified-memory question (16 vs 48/64/96 GB) and the processor question, without inducing memory exhaustion.
It reuses the scaling machinery (supervisor/worker, strict placement, numeric gate, watchdog, per-cell and total time limits) and adds a fixed grid
plus a resident-model probe. It is a separate design: nothing in the adaptive crossover search changed.

## What earlier benchmarks already cover (not repeated)
| Question | Where | Status |
|---|---|---|
| Sequential CPU vs GPU crossover (steps, batch 128/512/1024), large tier, P=4 | findings 1, 5-7 | measured, confirmed |
| Homogeneous batched CPU vs GPU at P=4, 8 (batch 128, 512) | findings 2, 5 | measured; P=8 memory 1.16-1.77 GB (session-dependent) |
| Heterogeneous batching coverage (20-29 %) | findings 2 | measured |
| CPU multiprocessing, 2 workers (1 thread each) | findings 2 | measured (0.65-0.78x of sequential) |
| Time split (construct/trace/first_exec/train/teardown) | findings 3 | measured |
| Memory growth across many individuals (clear_session/gc policies) | `--tests memdiag` | measured |
| P=16, P=32 batched; 4-worker multiprocessing | findings 4 | **skipped for headroom, never measured** |
| Memory per simultaneously resident model, fixed runtime overhead | - | **new** |
| Whether sequential population size changes resident memory | - | **new** (P=4..32 sequential, one model alive at a time) |
| Memory returned after release (cleanup inefficiency) | - | **new** (resident probe, release phase) |

## Design
1. **Resident probe** (one fresh process per device x tier): build and KEEP 1, 2, 4, 8, 16, 32 large-tier models (own optimiser, own traced step), run one step on every
   resident model after each rung, sample system memory (available-memory delta, which also sees Metal buffers), then release and sample after dropping references, after `gc`, and 3 s later.
   Gives fixed runtime memory, GB per model, bytes/parameter against the 16 B/parameter theory, per-model build time, per-step time vs number of resident models, and the fraction returned.
   *Safety:* before every rung its memory is projected from the growth measured so far (never below theory, x1.3 on the measured slope, x3 theory for the first model) and the rung is
   refused, with the reason recorded, if the projection exceeds 90 % of the headroom.
2. **Fixed grid** (tiers x steps x populations; defaults in the command below): sequential heterogeneous (CPU, GPU), homogeneous batched (CPU, GPU), CPU multiprocessing (2 and 4 workers).
   Run order: batched (the memory-hungry strategy), sequential, multiprocessing. Sequential keeps ONE model alive regardless of P; batched keeps P alive together; multiprocessing keeps one per worker.
   After the probe, a batched cell is skipped unless max(closed-form estimate, projection from the probe's measured bytes/parameter) x 1.5 fits the headroom.
3. **Four readings kept apart** (report section 3): memory capacity (peak vs headroom, memory-driven skips, probe stops), compute (does batched throughput still rise with P),
   TF-Metal/framework overhead (fixed runtime GB, per-step cost vs resident count), construction/cleanup (fixed share of wall time, build time per model, memory retained / returned).
4. **CALCULATED projection** (report section 4): resident models of the tested tier that fit a given amount of FREE memory if the measured linear growth continues. Memory only; no speed information.

## Safety
Existing limits unchanged and never raised: `--max-gb` (6 GB extra), 2 GB available floor, 80 % system use, 512 MB swap growth, 5 min per configuration, 20 min total.
Planning headroom is the smallest of the three memory limits given what is in use right now. The in-process watchdog and the supervisor backstop stop a block that crosses a limit.
Skipped, aborted and incomplete configurations are listed with reasons (report section 5) and are never counted as measurements.

## Command
```
python tf_memory_stress_test.py --tests scaling --scaling-design capacity --grappler none --scaling-tiers large --scaling-pops 4 8 16 32 --scaling-steps 20 --scaling-generations 2 --scaling-workers 2 4 --capacity-ladder 1 2 4 8 16 32 --capacity-resident-tiers large medium --scaling-max-total-min 20 --max-gb 6 --dry-run
```
(remove `--dry-run` to run). Output: `scaling/capacity_summary.md`, `scaling/cells.csv`, `report.json`.

## Limits of what this can show
One machine (M1 Pro, 16 GB). Memory per model is largely hardware-independent but is measured here only for this model family, optimiser and TensorFlow-Metal version.
A larger-memory machine can only be assessed for CAPACITY by calculation from the slope; whether its speed differs needs the same commands on that machine.
