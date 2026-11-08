# Memory retention study and real-world memory baseline

Two separate tools; neither raises a limit, and the monitor never touches another process.

## Phase 1 - retention and process lifecycle (`--scaling-design lifecycle`)
GPU-batched, homogeneous, large tier, P=4, 8, 16 repeated for 3 identical rounds:
1. **Long-lived process:** all 9 cells in ONE TensorFlow process. Snapshots per cell: before, right after execution, after `gc.collect()` (+1 s), settled (+3 s more).
   Each snapshot records system available memory, system use %, swap used, process RSS, process USS (private pages) and, on macOS, the OS `footprint` (best effort; the raw output
   of the first call is stored in the report so a wrong format assumption is visible). GPU-runtime memory is not exposed by TensorFlow-Metal; where `footprint` shows an `IOAccelerator` category
   it is recorded, otherwise "system minus process" is the only indirect indicator.
2. **Fresh process per cell:** the same 9 cells, each in a new process (same validation and warm-up). The supervisor records available memory and swap before the spawn and 2 s after the process
   exited, i.e. what the OS still counts as held after process termination.
3. **Medium-tier overhead probes** (fresh process each, 1..32 resident models): full build+trace+execute (heterogeneous), the same with one shared architecture, trace only, variables + optimiser only.
   Differences between them locate the ~85 MB/model overhead: variables, graph, kernel/pipeline state, or per-shape state.
Verdict rules (stated in the report): *stabilised* if the last round added <= max(0.15 GB, 10 % of the first round) - consistent with allocator caching / reuse; *still growing* if each round adds about
the same; a *leak* is not claimed unless growth continues over identical rounds AND the memory is not returned when the process exits.
Safety: the long-lived sequence stops itself when the memory held since the runtime was ready exceeds 60 % of the headroom, or the next cell's estimate x1.5 would not fit; limits and watchdog unchanged.

```
python tf_memory_stress_test.py --tests scaling --scaling-design lifecycle --grappler none --scaling-steps 20 --scaling-generations 2 --scaling-max-total-min 20 --max-gb 6 --dry-run
```
(remove `--dry-run` to run; ~10-15 min). Output: `scaling/lifecycle_summary.md`.

## Phase 2 - real-world baseline (`tools/memwatch.py`)
Read-only recorder; start and stop it yourself.
```
python tools/memwatch.py record --scenario A --note "usual apps, no ML"
python tools/memwatch.py record --scenario B --note "TensorFlow work" --track python
python tools/memwatch.py record --scenario C --note "ML + Lightroom + Photoshop" --track Lightroom Photoshop python
python tools/memwatch.py compare memwatch_results/*_A_*.csv memwatch_results/*_B_*.csv memwatch_results/*_C_*.csv
```
Per sample (default every 5 s, minimum 1 s): available memory, used %, wired/active/inactive/compressed memory, *demand* (wired+active+compressed), swap used and growth, swap-in/out rates,
macOS memory-pressure level (1 normal / 2 warning / 4 critical), memory-free %, CPU %, load, scheduler lag and spawn latency (responsiveness indicators), RSS of the process groups named with `--track`.
Every 60 s the top process groups by RSS go to `*.procs.jsonl`. Type a line + Enter at any time to add a marker (e.g. "export started"). Disk: buffered, flushed once a minute, ~3 MB/day.
Stop with Ctrl-C (also SIGTERM or `--max-minutes`, default 240); a killed run leaves a valid CSV and `summarize` rebuilds the summary.

Suggested use: start the monitor first and wait one minute (it uses that minute as the starting level), then begin the activity. For scenario C use a *moderate* ML workload (for example a
single-process training run you normally do), not the stress benchmarks; do not start benchmarks while the pressure level is not 1.

`compare` reports apps+macOS (A), the ML increase over its own start (B), the combined demand (C), the additive expectation and the difference (interference), and what C did to swap and pressure.
It deliberately does not compute a RAM recommendation: whether 16 GB is enough for C is decided by swap growth and non-normal pressure, and any shortfall is read from them.
