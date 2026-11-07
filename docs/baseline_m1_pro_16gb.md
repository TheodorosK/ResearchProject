# Baseline: MacBook Pro 14" (2021), Apple M1 Pro, 16 GB

Two full sessions of the suite on the same machine, one day apart. **Provenance:** the numbers below are copied from the
`summary.md` of run `20261106T233145Z` ("day 1") and the `tools/compare_reports.py` output for `20261107T065743Z`
("day 2"), both produced on the machine itself. The raw reports are in `bench_results/` on that Mac (git-ignored).
Nothing here is extrapolated to other hardware.

## Setup
| | |
|---|---|
| Machine | MacBook Pro 14" (2021), Apple M1 Pro (8 performance + 2 efficiency cores, 16 GPU cores), 16 GB unified memory |
| Power | AC adapter connected, Low Power Mode off |
| Software | macOS 27.0.1, Python 3.11.14, TensorFlow 2.15.1, Keras 2.15.0, tensorflow-metal 1.2.0, numpy 1.26.4 |
| Command | `python tf_memory_stress_test.py --tests performance neuroevolution memdiag --max-gb 6 --grappler none` |
| Fixed | seed 1234, `SimpleAdam` optimizer, strict GPU placement (no CPU fallback), float32, graph mode, batch 256 |
| Differences between the days | day 1: `--ne-cleanup generation`, memdiag policies none+gc+clear; day 2: `--ne-cleanup gc`, memdiag none+gc (defaults changed in between) |

**`--grappler none` is required** on this setup: with the default graph optimizer the fused `MatMul+BiasAdd+ReLU` computes wrong
values on the GPU (see README, *Known issue*). All numbers below are valid only with that setting.

## Results (steady state, median of 5 repeats x 100 steps)
| Workload | Device | Day 1 samples/s | Day 2 samples/s | Change | ms/step (d1 / d2) |
|---|---|---|---|---|---|
| MLP 784-4x2048-10 (14.2 M params) | CPU | 4,952 | 4,655 | -6.0 % | 51.7 / 55.0 |
| | GPU | 15,872 | 15,740 | -0.8 % | 16.1 / 16.3 |
| Convnet (545 k params, 32x32x3) | CPU | 5,395 | 5,330 | -1.2 % | 47.5 / 48.0 |
| | GPU | 26,784 | 27,393 | +2.3 % | 9.56 / 9.35 |

| Ratio (calculated within each run) | Day 1 | Day 2 |
|---|---|---|
| GPU speed-up, MLP | 3.21x | 3.38x |
| GPU speed-up, convnet | 4.96x | 5.14x |
| Neuroevolution GPU/CPU throughput | 0.108x | 0.116x |

In-session CV of the per-repeat time: 0.2 % - 3.1 %. Day-to-day spread of throughput: 0.8 % - 6.0 %.

## Neuroevolution (population 12, 5 generations, 52 evaluations, synthetic regression)
| | Day 1 | Day 2 |
|---|---|---|
| CPU individuals/s | 5.20 | 5.25 |
| GPU individuals/s | 0.563 | 0.609 |
| Best test R2 (CPU / GPU) | 0.925 / 0.928 | 0.925 / 0.928 |
| Peak extra system memory CPU / GPU | 0.17 / 1.11 GB | 0.18 / 0.95 GB |

Per-generation best validation R2 was identical on both days (the GA is deterministic) and CPU and GPU agreed in the early
generations. Tiny evolved networks are dominated by per-model start-up cost: the GPU pays roughly 1.4-1.8 s per new model
versus about 0.15 s on the CPU.

## Memory
* No swap activity in any job; minimum available memory was 4.6 GB (day 1); peak extra system memory <= 1.11 GB.
* `memdiag` (GPU, identical build/train/discard cycles): host RSS grows ~6-10 MB per model; the rate was reproducible
  (`none` 10.4 and 10.4 MB/cycle on the two days; `gc` 9.5 and 8.4). Over 120 cycles `gc` showed slowing growth
  (RSS slope 8.0, 7.0, 2.6 MB/cycle by window); `none` showed no clear slowing (8.9, 2.9, 7.5).
* `tf.keras.backend.clear_session()` after every cycle drove system memory up ~110-160 MB per cycle and hit the 80 % cap
  after 17 cycles (one run) - do not call it in a loop on this setup. Its effect on neuroevolution per-generation growth is NOT
  established (two `gc` runs on different days gave 135 and 27 MB/gen RSS growth).

## Reading the spread
* Throughput and CPU/GPU ratios can be trusted to about +-6 % between days; treat smaller differences as noise.
* Compile / first-execution time (GPU convnet 1.34 s -> 0.36 s), per-generation memory growth, peak memory and CV varied much more
  and are marked *volatile* in `tools/compare_reports.py`. A warm vs cold Metal shader cache is a plausible but untested cause;
  compare second runs with second runs.
* One repeat is a small sample; more days would tighten these numbers.

## Not measured
Thermal throttling (pmset returned no data), GPU utilisation / clocks, models much larger than 14 M parameters, data-pipeline-bound
training, transformers, and any other chip or memory size. Comparing with another machine requires running the same command there
(same TensorFlow / tensorflow-metal versions, `--grappler none`, same seed) and `python tools/compare_reports.py <this report> <new report>`.
