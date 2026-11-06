#!/usr/bin/env python3
"""TensorFlow memory-stress / performance / neuroevolution benchmark for macOS (Apple Silicon).

Legacy tests (unchanged CLI):   alloc  train  leak
New tests:                      performance  neuroevolution  memdiag

    python tf_memory_stress_test.py --tests performance neuroevolution --quick --dry-run
    python tf_memory_stress_test.py --tests performance --max-gb 6
    python tf_memory_stress_test.py --tests neuroevolution --max-gb 6
    python tf_memory_stress_test.py --tests performance neuroevolution --max-gb 6

See README_tfbench.md.  Run with -h for all options.
"""
import sys

from tfbench.cli import main

if __name__ == "__main__":
    sys.exit(main())
