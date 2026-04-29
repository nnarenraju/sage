#!/usr/bin/env python3
"""
MLGWSC-1 evaluation wrapper for the O3b Sage results.

Calls sage/benchmark/mlgwsc1/evaluator.py with the correct paths for the
testing_month_D4_seeded dataset.  Results go to run_export/benchmark-mlgwsc1/.

Run from the runs/o3b directory:
    python3 evaluate_mlgwsc1.py
"""

import os
import sys

RUN_DIR  = os.path.dirname(os.path.abspath(__file__))
SAGE_DIR = os.path.join(RUN_DIR, "..", "..")
sys.path.insert(0, RUN_DIR)
sys.path.insert(0, SAGE_DIR)

# Config must be set before importing sage modules.
from config import set_configs
set_configs()

from sage.benchmark.mlgwsc1.evaluator import main as evaluator_main

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTING_DIR   = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
ORCHID_DIR    = "/local/scratch/igr/nnarenraju/orchid_data/results"
BENCHMARK_DIR = os.path.join(RUN_DIR, "run_export", "benchmark-mlgwsc1")
EVAL_DIR      = os.path.join(BENCHMARK_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

DATASET           = 4
DURATION_SECONDS  = 2592000.0   # 30-day testing dataset

# ---------------------------------------------------------------------------
# Evaluator arguments
# ---------------------------------------------------------------------------

raw_args = [
    "--injection-file",     os.path.join(TESTING_DIR, "injections.hdf"),
    "--foreground-events",  os.path.join(BENCHMARK_DIR, "fg_events.hdf"),
    "--foreground-files",   os.path.join(TESTING_DIR,  "foreground.hdf"),
    "--background-events",  os.path.join(BENCHMARK_DIR, "bg_events.hdf"),
    "--output-file",        os.path.join(EVAL_DIR, "eval.hdf"),
    "--output-dir",         EVAL_DIR,
    "--orchid-results",     ORCHID_DIR,
    "--far-scaling-factor", str(DURATION_SECONDS),
    "--dataset",            str(DATASET),
    "--team1",              "Sage",
    "--team2",              "PyCBC",
    "--verbose",
    "--force",
]

evaluator_main(
    raw_args,
    cfg_far_scaling_factor=DURATION_SECONDS,
    dataset=DATASET,
)
