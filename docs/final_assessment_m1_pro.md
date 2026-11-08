# Final assessment: MacBook Pro 14" (2021), Apple M1 Pro, 16 GB - and what it means for the next purchase

Scope: consolidates every measurement in `docs/findings_neuroevolution_m1_pro_16gb.md`, `docs/baseline_m1_pro_16gb.md`, `docs/capacity_benchmark.md` and `docs/memory_baseline_protocol.md`.
No new measurements were made for this document. Stack: macOS 27.0.1, Python 3.11, TensorFlow 2.15.1, tensorflow-metal 1.2.0, `--grappler none`, SimpleAdam, strict GPU placement. One machine, one software stack, mostly single runs
(confidence intervals only where stated), synthetic regression data (4,096 x 32 inputs).

Labels: **[C] confirmed** = measured, replicated or with a 95 % interval that excludes the alternative | **[I] inference** = follows from measurements but was not tested directly | **[U] unresolved**.

## 1. CPU performance
* **[C]** Standalone MLP / convnet training on the CPU: 4,952 and 4,655 / 5,395 and 5,330 samples/s on two days (day-to-day spread <= ~6 %).
* **[C]** For neuroevolution the CPU is compute-bound: its per-step cost grows with batch size (large tier ~27 ms at batch 512, ~39 ms at 1024) and 92-94 % of its time is training.
* **[C]** CPU wins for small and medium networks (GPU/CPU 0.15-0.21 at every tested step count), for short training, and for heterogeneous populations at batch <= 128.
* **[C]** CPU batching of identical models gives nothing (1.00-1.03x of sequential; ~3 individuals/s plateau from P=8). Two worker processes with one thread each are *slower* than sequential (0.67-0.93x).
* **[U]** Four or more workers, more threads per worker, and any benefit from more performance cores: never measured (4-worker runs were skipped for memory headroom).

## 2. GPU performance
* **[C]** Standalone MLP / convnet training: GPU 3.2-3.4x / 5.0-5.1x faster than the CPU (15,872 and 15,740 / 26,784 and 27,393 samples/s).
* **[C]** Neuroevolution, sequential evaluation, large tier (GPU/CPU throughput, 95 % CI): batch 128 - 0.654 [0.610, 0.701] at 50 steps, parity only near 200 steps; batch 512 - 0.673 [0.659, 0.687] at 20 steps, 1.22 [1.17, 1.27] at 50, 2.05 [2.01, 2.09] at 200 (**confirmed crossover between 20 and 50 steps, ~37**); batch 1024 - 0.964 [0.875, 1.06] at 20 (parity), 1.67 [1.50, 1.85] at 50, 2.46 [2.33, 2.60] at 200.
* **[C]** Homogeneous populations evaluated as one batch (batch 128, 20 steps): GPU-batched / CPU-batched 0.871 [0.841, 0.902] at P=4 (CPU faster) -> 1.25 [1.21, 1.29] at P=8 (**confirmed crossover**) -> 1.73 [1.64, 1.82] at P=8, 50 steps; 1.70 [1.65, 1.75] at P=4 with batch 512. GPU-batched is 2.0-4.0x faster than GPU-sequential.
* **[C]** Heterogeneous (random-architecture) populations barely batch (20-29 % of individuals, ~1.0-1.1x).
* **[I]** The estimated sequential break-even falls with batch size (~37 steps at 512, ~22 at 1024, labelled estimates from a fitted cost model; the fit predicted the later batch-512 results within ~10 %). Break-evens beyond the tested range must not be read as measurements.
* **[U]** GPU-batched beyond P=16 (P=32 never run); whether the GPU would keep scaling is open. At P=16 it was still rising (x1.29 for the last doubling, single run).

## 3. TensorFlow-Metal overhead
* **[C]** Per-step GPU cost is flat at ~8-10 ms from batch 128 to 1024: the sequential GPU path is limited by per-step framework/dispatch cost, not by GPU arithmetic or memory.
* **[C]** Per-model fixed cost on the GPU is ~0.8 s (construct+trace ~15 %, first execution 27-52 %, teardown 10-20 %): ~80 % of wall time at <= 20 steps, vs 37-40 % on the CPU.
* **[C]** Correctness hazard: with the default graph optimiser the fused MatMul+BiasAdd+ReLU gives wrong results on tensorflow-metal 1.2.0 / macOS 27.0.1 (23 %/7 % loss differences). `--grappler none` fixes it; the numeric gate refuses GPU timings otherwise. Other workarounds were needed (no Keras optimiser `iterations` variable, no Elu, host-side weight init).
* **[C]** The first execution of each separately-held training function on the GPU costs ~90 MB of process memory in the medium tier (variables and tracing cost ~0); identical architectures cost the same.
* **[I]** These overheads belong to TensorFlow-Metal, not to Apple GPUs in general; other stacks (PyTorch MPS, JAX, MLX) were not tested and could behave differently.
* **[U]** Why the large tier shows only ~28 MB per executed function; whether sharing one traced function between same-architecture models removes the ~90 MB.

## 4. Memory capacity
* **[C]** Resident model state is weights + two Adam moments = 12 B/parameter (32 resident large-tier models: 1.6 GB on the GPU, 1.0 GB on the CPU; GPU runtime fixed cost 0.4-0.7 GB).
* **[C]** Peak additional memory of every cell that ran: sequential 0.1-0.6 GB regardless of population size (P=4..32; one model alive at a time); GPU-batched 1.0-1.8 GB (P=4..16); CPU-batched 1.97 GB at P=32. Swap growth was 0 MB in every benchmark.
* **[C]** Nothing measured needed more than ~2 GB for a single configuration. Skipped configurations (GPU-batched P=32, 4 workers, P=16 in the last run) were skipped because of the headroom this machine had under its normal load (2.8-4.3 GB, with a 1.5x safety margin), not because they were measured to exceed 16 GB.
* **[C]** In daily use macOS plus your login apps demand ~7.4-8.7 GB of the 16 GB (memwatch; psutil "used" 9.6 GB during Spotlight indexing); the benchmark tool's 80 % ceiling therefore leaves only ~3-4 GB.
* **[I]** A design that keeps hundreds of separately executed small models alive on the GPU could need tens of GB (~90 MB each); batching or releasing models avoids it. Real datasets and far larger networks than the 4 M-parameter large tier would add memory that was **not** tested at all.

## 5. Memory retention
* **[C]** Long-lived GPU process, P=4/8, three identical rounds: held memory plateaued at ~0.5 GB (system-wide) after the first round and added +0.01 GB later (stabilised); OS footprint 0.41 -> 2.38 -> 2.44 GB, also a plateau. Fresh process per cell: memory returned on exit (net -0.04 to +0.10 GB). Throughput did not drift.
* **[C]** RSS and USS understate the GPU runtime's memory (footprint +2.0 GB vs RSS +0.3, USS +0.1, system available -0.5 for the same cell).
* **[I]** Consistent with allocator/pool caching, not a leak. An earlier "~3.9 GB accumulates across cells" reading was not reproduced.
* **[U]** P=16 under the controlled protocol; why footprint growth is not fully visible as lost available memory.

## 6. Scalability
* **[C]** Sequential: time grows linearly with population (throughput flat in P). GPU-batched: throughput rises with P (2.34 -> 3.79 -> 4.88 ind/s at P=4/8/16, single run). CPU: no scaling with P or processes as tested.
* **[I]** The M1 Pro GPU was not saturated at P=16 in the batched path - the only measured situation where more GPU capacity could plausibly help.

## 7. Practical limitations of this evidence
Single machine and stack; synthetic data; many single runs; confidence intervals only for the sequential/batched ratios; the memory monitor was started late in the one concurrent recording; Lightroom Classic and Photoshop were never measured (scenario C absent); no newer Apple chip was measured and none is inferred; memory metrics other than the 0.5 s sampler are noisy at the +-0.2 GB level.

## 8. Purchase assessment (evidence-informed, not measured on new hardware)
### 8.1 Memory: 48 vs 64 vs 96 GB
What the data say: the ML workloads as tested need a few GB at most; retained memory is bounded and returned; the machine's pain is OS + apps (~8-10 GB demand on 16 GB) and the absence of headroom for everything else.
What they cannot say: how much Lightroom Classic and Photoshop need next to a TensorFlow job (not measured), how large your real datasets and models will be, and which engine design you will build.

| Capacity | What the evidence supports | Judgement |
|---|---|---|
| 48 GB | ~3x the memory of today. Covers every measured configuration with a very wide margin and leaves ~30+ GB for macOS, Lightroom/Photoshop and several concurrent ML jobs. | **Best supported by the data** as sufficient for the measured workloads plus a normal creative-app load. |
| 64 GB | Extra ~16 GB over 48. No measured workload needs it. It becomes justified only if you expect large in-memory datasets, networks far above the tested 4 M-parameter tier, resident-population designs (hundreds of live GPU models), or very heavy Photoshop/Lightroom sessions in parallel. | A **risk-margin / longevity choice**, not an evidence-based need. Reasonable if the price gap is small relative to the machine's 4-5-year life, because unified memory cannot be upgraded. |
| 96 GB | Nothing measured or foreseeable from these tests approaches it. | **Not supported** by any evidence here; only a speculative future-proofing bet. |

Practical reading: if the cost of 64 over 48 is small, 64 buys margin against the untested items above; if it is large, 48 is the evidence-supported choice. 96 is not justified by this work.
Caveat on feasibility (not measured here): which RAM sizes are offered with which chip tier differs per generation, and in recent generations the higher capacities have been tied to Max-class chips - verify the current configurator before treating "48 vs 64 vs 96" and "Pro vs Max" as independent choices.

### 8.2 Processor: Pro vs Max class
* **Evidence for caring about GPU capacity:** only the batched path (homogeneous populations) kept scaling with more work in flight (not saturated at P=16) and showed the largest GPU advantage (1.6-4x). If the engine is designed around stacked/batched evaluation, GPU throughput and memory bandwidth are plausible levers - a Max-class GPU might help, but **this was not measured**.
* **Evidence against paying for GPU cores for the sequential path:** the sequential GPU path is overhead-bound (flat ~10 ms/step from batch 128 to 1024, ~80 % fixed per-model cost). More GPU cores would not obviously speed it up; faster single-thread CPU/dispatch and better software would. Under TensorFlow-Metal the CPU is also faster for small/medium networks and short training.
* **CPU side:** the CPU path is compute-bound and single-stream in all measurements; whether more performance cores help (multiprocessing with >= 4 workers) is unmeasured. Core counts differ by generation and tier - check them rather than assume.
* **Outside this study:** Lightroom/Photoshop GPU use and any other framework were not measured, so they cannot be used to separate Pro from Max here.
* **Judgement:** a Pro-class chip is sufficient for everything measured; a Max-class chip is justified by (a) needing a memory capacity only it offers, (b) a batched-population engine design where GPU scale is expected to matter, or (c) creative-app needs outside this study - not by the sequential results.

### 8.3 Engine-design guidance that follows from the data (hardware-independent)
Prefer stacked/batched evaluation of same-architecture populations on the GPU; use the GPU sequentially only for large networks, batch >= 512 and >= ~40 steps; keep CPU for small/medium networks, heterogeneous topologies at small batch and short training; do not keep many separately executed GPU functions alive; use a fresh process per generation or batch if you want a hard bound on retained memory; re-run the numeric gate on any new macOS/TensorFlow-Metal version.

### 8.4 What would change this assessment
Scenario C showing swap growth or non-normal memory pressure with Lightroom + Photoshop + a modest TensorFlow job (-> more memory); real datasets/models in the tens of GB (-> 64 GB or more); an engine design with many resident GPU models (-> more memory or a redesign); evidence that your engine is batched and GPU-bound (-> Max-class GPU); a different framework with different overhead (-> revisit the GPU conclusions).

### 8.5 If you want a cheap, optional check (no benchmark)
Run `tools/memwatch.py` during a normal Lightroom + Photoshop session for a day (scenario A/C); it only observes. Skip it if you are happy to decide on the above.
