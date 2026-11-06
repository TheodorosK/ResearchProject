# tfbench - TensorFlow performance, neuroevolution and memory benchmark (macOS / Apple Silicon)

A reproducible baseline for deciding whether a Mac is enough for TensorFlow / neuroevolution work, and for
comparing it later with other machines (e.g. 32-128 GB unified memory, different chips).
It measures; it does not predict. **It makes no claim about hardware you have not run it on.**

| Test | What it answers |
|---|---|
| `performance` | How fast is *controlled* training on CPU vs the Metal GPU? (phases timed separately, device placement verified) |
| `neuroevolution` | How does a realistic GA + Keras workload behave: throughput, time/generation, memory per generation? |
| `memdiag` | Does memory growth across build/train/discard cycles look like caching, retained references, or a native leak? |
| `alloc`, `train`, `leak` | Original stress tests, CLI unchanged. |

## Layout

```
tf_memory_stress_test.py      entry point (thin)
tfbench/                      limits, genetics, perf, evolution, report, cli, legacy ...
tests/                        unit tests (genetic operators, resource limits, reports, CLI)
docs/example_report/          example output (generated on a Linux CPU sandbox - structure only)
README_tfbench.md             this file
```

The suite is now several files: **clone the repository** (a single `curl` of one file no longer works).

## Install (your existing `ml-tf` conda env)

```bash
git clone -b claude/optimistic-galileo-6jk5us https://github.com/TheodorosK/ResearchProject.git
cd ResearchProject
conda activate ml-tf
pip install psutil          # the only extra dependency (numpy/tensorflow are already there)
```

Already cloned? `git fetch origin claude/optimistic-galileo-6jk5us && git checkout claude/optimistic-galileo-6jk5us`.

## Run

```bash
python -m unittest discover -s tests -t .            # 1. unit tests first (TensorFlow smoke tests included)
python tf_memory_stress_test.py --tests performance neuroevolution memdiag --quick --dry-run   # 2. validate + estimate, trains nothing
```

The three requested forms (full configuration, 6 GB cap):

```bash
python tf_memory_stress_test.py --tests performance --max-gb 6
python tf_memory_stress_test.py --tests neuroevolution --max-gb 6
python tf_memory_stress_test.py --tests performance neuroevolution --max-gb 6
```

### Recommended first run on the M1 Pro / 16 GB (conservative)

Plug in the charger, close browsers/IDEs/VMs, then:

```bash
python tf_memory_stress_test.py --tests performance neuroevolution --quick --max-gb 4
python tf_memory_stress_test.py --tests memdiag --quick
```

`--quick` = small workloads (MLP only, 3 repeats x 30 steps; population 6 x 3 generations; 10 diagnostic cycles).
Explicit flags always win over `--quick`. If those look sane, run the full versions above, then repeat the
whole thing on a different day / after a reboot - single sessions say nothing about session-to-session variance.
Add `--tests ... --ne-workers 2` once to see (not assume) whether parallel evaluation helps.

Results go to `bench_results/<UTC timestamp>/`: `report.json`, `measurements.csv`, `summary.md`
(plus `raw/` per-job files). Exit code: 0 = all jobs completed, 1 = some did not, 2 = invalid config / refused to start.

## Safety limits

| Limit | Flag | Default |
|---|---|---|
| Extra system memory vs the job's baseline | `--max-gb` | min(6, 50 % RAM) |
| Minimum available memory | `--min-avail-gb` | 2 |
| Maximum system memory use | `--max-sys-pct` | 80 |
| Swap growth | `--max-swap-mb` | 512 |
| Runtime per benchmark (all jobs of a test share it) | `--max-runtime-min` | 20 |
| Generation cap | `--max-generations` (vs `--ne-generations`, default 5) | 10 |

* The *effective* cap is whichever limit bites first: with 5 GB available and a 2 GB floor, at most ~3 GB can be used even though `--max-gb` is 6.
* **Limits are never raised automatically.** If the machine already violates them the run refuses to start
  (exit 2) and says why; a configuration whose estimate exceeds a limit is rejected, not shrunk silently.
* Every job runs in its **own process**. Inside it a background watchdog samples the OS every 0.5 s; timed loops
  only read an `Event` flag (no psutil call, no Keras callback). On a violation the job stops at its next check,
  flushes partial results and is reported as `aborted` with the reason.
* **Limitation:** a native call blocked inside TensorFlow/Metal cannot be interrupted from Python. Backstops: the
  worker exits after 30 s of unacknowledged abort, and the supervisor SIGTERMs/SIGKILLs a worker that stays over a
  limit or its time budget. Ctrl-C terminates the child and still writes a report.
* The legacy `alloc/train/leak` tests keep their behaviour but now use these (tighter) defaults; the old
  `--max-swap-gb` still works as an alias.
* `--dry-run` validates everything, prints analytic estimates (clearly labelled *estimated*) and trains nothing.

## What `performance` does

Per (device, workload) it launches a fresh process (so CPU runs hide the GPU before TensorFlow initialises):

1. fixed seeds, fixed batch/steps/optimizer (Adam, same class on both devices), float32, `jit_compile=False`;
   synthetic data generated locally; CPU and GPU receive identical data and the same weight-initialiser seeds (the first-step-loss comparison below checks this);
2. **separately timed phases:** data generation, data transfer, model build, optimizer init, graph *trace*,
   *first execution* (compilation + first step), *warm-up*; then
3. **steady state:** `--perf-repeats` repeats of `--perf-steps` steps; each repeat ends with a host read-back of the
   loss and a weight (a real completion barrier on the GPU). Reported: median, mean, stdev, CV, min/max of the
   per-repeat time, steps/s, samples/s, and CPU-vs-GPU speed-up (from both medians).
4. **device placement is verified, not assumed:** a probe op, every model variable and the input/output tensors
   must report the requested device, otherwise no timings are reported. `--placement`:
   `strict` = soft placement off (an op without a GPU kernel raises -> reported as an unsupported op),
   `soft` = TensorFlow may run such ops on the CPU (every result is labelled `placement=soft`),
   `auto` (default) = strict first; if strict fails on an unsupported op, retry **once** in soft mode, labelled and noted.
   A missing GPU is `unsupported` - there is **never** a silent CPU fallback.
5. RSS peak, sampled system-memory peaks, swap growth/page-outs, `pmset` thermal indicators before/after every
   repeat, and a first-vs-last-third drift ratio (> 1.10 is flagged as *possible* throttling or background load).
6. Execution mode is recorded (`graph`: `tf.function`; `eager` with `--perf-mode eager`).

The first-step losses of CPU and GPU are compared as a sanity check that the workloads really are comparable.

## What `neuroevolution` does

* **Genome:** 1..N hidden layers (units, activation), learning rate, weight-mutation sigma/rate, architecture-mutation rate.
* **Operators:** tournament selection; elitism (elites are carried over unchanged and not re-evaluated);
  architecture mutation (add/remove layer, resize, change activation - only operators legal for the current depth);
  log-normal numeric mutation; crossover. Every operator ends with `repair`, so architectures are always valid
  (fuzz-tested over thousands of operations).
* **Crossover of incompatible architectures:** genomes are *compatible* when their depths match -> layer-wise uniform
  crossover. Otherwise the child keeps the depth of a randomly chosen primary parent and, per position, takes the
  primary's layer or (p = 0.5) the secondary's layer at the proportionally aligned position. Numeric genes are blended.
* **Optional weight inheritance** (`--ne-inherit-weights`): overlapping blocks of parent Dense tensors are copied
  into the child's fresh initialisation (hidden layers by index; the output layer only if depths are equal), then
  perturbed using the genome's weight sigma/rate. Children then train *cumulatively*, so fitness is not comparable
  with a non-inheriting run (the report says so).
* **Fitness protocol:** synthetic nonlinear regression with noise. Selection uses **validation R²**; training R² is
  logged separately; the **test split is evaluated exactly once**, for the best-by-validation individual, after
  evolution. Every individual gets the same budget (`--ne-epochs` x floor(`--ne-train-n` / `--ne-batch`) optimiser steps, run in an explicit-device `tf.function` loop - the same mechanism as `performance`, so unsupported GPU ops surface in strict mode), seeds derive from (seed, uid).
* **Per generation:** evaluations, failures, wall time, individuals/s, fit time, best/mean R², process cores busy,
  system CPU %, RSS and system memory before/after cleanup, peak, memory remaining, swap growth.
* **Parallelism** (`--ne-workers 2..4`): threads in one TensorFlow runtime. A sequential-vs-parallel probe on the
  same individuals measures the speed-up instead of assuming it. All models share one Metal GPU, so a speed-up is not expected.
* **Cleanup policy** (`--ne-cleanup each|generation|never`): `clear_session()` + `gc.collect()` timing is recorded.
* **`memdiag`:** builds/trains/discards the same model N times under policies `none`, `gc`, `clear`, each in a
  separate process, tracking RSS, system-available memory, `tracemalloc` (Python-level) and GC object counts.
  Verdicts: `bounded` / `saturating` (allocator-cache-like), `python_retention` (references held),
  `continued_native_growth` (**potential** native leak or unbounded cache - confirm with more `--memdiag-cycles`).
  Rising RSS alone is never called a leak.

## Report contract

* `report.json` (strict JSON, `schema_version` 1): `hardware`, `software`, `limits`, `config`, `jobs[]` (each with
  `status`, `reason`, `measurements[]`, `data`, `notes`), `derived`, `incomplete`, `notes`, `limitations`.
* `measurements.csv`: `job,test,device,workload,placement,name,value,unit,kind,repeat,generation,cycle,policy`.
* `summary.md`: sections 1-10 (configuration, devices, CPU/GPU performance, memory, swap, neuroevolution throughput,
  time per generation, memory growth, incomplete tests, limitations).
* Every value has a `kind`: **measured** (clock/OS/TensorFlow), **calculated** (arithmetic on measured values),
  **estimated** (analytic model, not observed). See `docs/example_report/` for the structure.

## tensorflow-metal constraints to keep in mind

* Use the tensorflow-metal release Apple documents for your TensorFlow version; the report records both versions.
* No public API for GPU utilisation or GPU memory (`get_memory_info` is unsupported): GPU memory only shows up as
  a drop in system-available (unified) memory. `powermetrics` would need `sudo` and is not used.
* The default optimizer is `SimpleAdam` (Keras-Adam update rule, float32 state only; ~30 % slower than Keras' fused Adam on CPU in my tests, so absolute samples/s depend on the optimizer) because Keras optimizers keep an int64
  step counter that tensorflow-metal cannot place on the GPU - that forces soft placement. `--optimizer keras` selects the Keras
  legacy Adam instead (expect `placement=soft` on Metal). The same optimizer is used on CPU and GPU and is recorded in the report.
* `Elu` has no GPU kernel in tensorflow-metal (observed: `Could not satisfy device specification ... [Op:Elu]`), so the default
  neuroevolution activation space is `relu tanh sigmoid swish` for both devices (`--ne-activations` overrides; adding `elu` forces `placement=soft`).
* **Numerical validation gate:** before any timing, `performance` (MLP) and `neuroevolution` compare the eager AND
  `tf.function` forward-pass loss with a float64 host calculation from the same weights and batch. A job that disagrees by
  more than 1e-3 is marked `failed` and reports no timings/fitness (`--skip-numeric-gate` disables this - not recommended).
  Observed on this suite's author's M1 Pro: eager forward exact, graph-mode forward of Keras Dense layers wrong on the GPU.
  `--grappler no-remapper|none` applies a graph-optimizer workaround to every job; the numerics tool prints a matrix showing
  which setting restores correct results.
* **Known issue (observed, macOS 27.0.1 / TF 2.15.1 / M1 Pro; plugin version is in the report):** in `tf.function` graph mode
  the fused `MatMul + BiasAdd + ReLU` computes wrong values on the GPU (100 % error in the isolated op; eager mode and
  `BiasAdd` alone or with `tanh` are exact). `--grappler no-remapper` does NOT fix it (so it is not TensorFlow's remapper);
  `--grappler none` (all graph rewriting off) does. Use `--grappler none` for all jobs on such a setup; absolute throughput then
  reflects un-optimised graphs on both devices. Keras `fit()` runs in graph mode, so check your own models too.
* `tools/metal_numerics_check.py` compares individual ops on CPU, GPU and a float64 NumPy reference - run it if CPU and GPU results disagree.
* XLA (`jit_compile`) is not used; float64 and some ops lack Metal kernels -> reported as unsupported in `strict` mode.
* GPU reductions are not bit-reproducible; seeds fix data, initialisation and the GA lineage, not every last bit.

## Limitations

See section 10 of every `summary.md` (generated from `tfbench/report.py::LIMITATIONS`). In short: one machine and one
session; coarse OS memory counters; RSS excludes GPU buffers; no per-op placement API; thermal telemetry is partial and
its absence proves nothing; blocked Metal calls can only be stopped by the supervisor; small synthetic workloads do not
represent every model; FLOP and pre-run memory numbers are estimates.

## Comparing machines later

Run the identical commands (same seed, TensorFlow and tensorflow-metal versions, plugged in, same `--quick`/full
settings) on each machine and compare `report.json` files field by field (`derived.cpu_vs_gpu_performance`,
`overall_individuals_per_s`, time per generation, peak memory, swap). Expect larger unified memory to matter only where
this suite shows memory pressure (swap growth, throughput collapse); it will not make a compute-bound job faster by itself.
