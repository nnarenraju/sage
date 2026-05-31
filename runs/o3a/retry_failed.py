#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retry_failed.py — CLI wrapper for retrying failed/missing O3a segment downloads.

Usage
-----
    python retry_failed.py                       # retry all detectors
    python retry_failed.py --detector H1         # retry H1 only
    python retry_failed.py --detector L1 V1      # retry L1 and V1
    python retry_failed.py --workers 8           # override worker count

All retry logic lives in sage.data.primer.retry.
"""

import sys
import argparse
from pathlib import Path

# --- repo root on path ---
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).parent))

from config import set_configs
from sage.data.primer.retry import retry_detector

_DATA_DIR = Path("/data/wiay/nnarenraju/data_release")
_RUN = "O3a"


def main():
    parser = argparse.ArgumentParser(description="Retry failed GWOSC segment downloads.")
    parser.add_argument(
        "--detector", nargs="+", default=["H1", "L1", "V1"],
        help="Detectors to retry (default: H1 L1 V1)",
    )
    parser.add_argument(
        "--run", default=_RUN,
        help=f"Observing run (default: {_RUN})",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel download workers (default: 8)",
    )
    args = parser.parse_args()

    set_configs()

    for det in args.detector:
        print(f"\n{'='*60}")
        print(f" Retrying {det} / {args.run}")
        print(f"{'='*60}")
        retry_detector(det, args.run, _DATA_DIR, num_workers=args.workers)


if __name__ == "__main__":
    main()
