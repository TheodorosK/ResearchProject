#!/usr/bin/env python3
"""Generate a SYNTHETIC scaling-report illustration (layout / crossover analysis demo).  Not a measurement of any machine.

    python tools/make_synthetic_scaling_example.py docs/example_scaling_synthetic
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tfbench.common import sanitize  # noqa: E402
from tfbench.scaling import report as SR, run as R, synthetic as SY, workload as W  # noqa: E402


def main(out_dir: str) -> None:
    st = W.Settings()
    clock = [0.0]
    job = R.run_scaling(st, 6.0, SY.make_runner(st, clock, seed=7), clock=lambda: clock[0], log=lambda *_: None)
    job["data"]["synthetic"] = True
    job["data"]["synthetic_truth"] = {f"{d}|{t}": {"fixed_s_per_model": f, "s_per_step": s} for (d, t), (f, s) in SY.TRUTH.items()}
    job["data"]["synthetic_true_break_even_steps"] = {t: SY.true_break_even(t) for t in W.TIER_ORDER}
    os.makedirs(out_dir, exist_ok=True)
    SR.write_artifacts(out_dir, job)
    with open(os.path.join(out_dir, "scaling", "scaling_summary.md"), "a") as fh:
        fh.write("\n### Synthetic ground truth (known only because this data was generated)\n\n```json\n"
                 + json.dumps(sanitize({"truth_seconds_per_model_and_per_step": job["data"]["synthetic_truth"],
                                        "true_break_even_steps": job["data"]["synthetic_true_break_even_steps"]}), indent=2) + "\n```\n")
    print("written", out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/example_scaling_synthetic")
