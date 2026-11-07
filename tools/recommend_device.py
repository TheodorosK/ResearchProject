#!/usr/bin/env python3
"""Recommend a neuroevolution execution strategy from a scaling report (RECOMMENDATION ONLY - changes nothing).

    python tools/recommend_device.py bench_results/RUN/report.json --params 5e6 --layers 6 --steps 50 --batch 128 --population 16
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tfbench.scaling import scheduler  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--params", type=float, required=True, help="parameters per network")
    ap.add_argument("--layers", type=float, required=True, help="hidden layers per network")
    ap.add_argument("--steps", type=int, required=True, help="training steps per individual")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--population", type=int, required=True)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--available-gb", type=float, default=None)
    ap.add_argument("--homogeneous", action="store_true", help="all individuals share one architecture (batchable)")
    a = ap.parse_args(argv)
    with open(a.report) as fh:
        rep = json.load(fh)
    job = next((j for j in rep["jobs"] if j.get("test") == "scaling"), None)
    if not job or not job.get("data", {}).get("analysis", {}).get("scheduler_model"):
        print("This report has no scaling calibration (run `--tests scaling` first).", file=sys.stderr)
        return 2
    an = job["data"]["analysis"]
    out = scheduler.recommend(an["scheduler_model"], params=a.params, layers=a.layers, steps=a.steps, batch=a.batch,
                              population=a.population, generations=a.generations, available_gb=a.available_gb,
                              homogeneous=a.homogeneous, validation=an.get("scheduler_validation"),
                              machine={"chip": rep.get("hardware", {}).get("chip")})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
