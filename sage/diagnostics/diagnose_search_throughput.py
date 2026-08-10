#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : diagnose_search_throughput.py
Description   : Measure scoring throughput and the cost of the two forward stages.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Background dominates the cost of a campaign, and how it is best computed depends on the
split between the per-detector stage and the shared stage. Caching the per-detector
stage lets each slide re-run only the shared part, which pays off in proportion to how
much of the work is per-detector. That split is measured here rather than assumed, and
the caching path is only usable when the per-detector stage is genuinely separable.

Reports the achieved rate, the split between stages, the number of slides beyond which
caching wins, and the resulting estimate for a campaign.

Usage
-----
    python -m sage.diagnostics.diagnose_search_throughput --config config_o4a_HL
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence


def measure_full(engine, reader, n_windows: int) -> dict:
    """Windows scored per second through the whole network."""
    raise NotImplementedError


def measure_split(engine, reader, n_windows: int) -> dict:
    """Cost of the per-detector stage and the shared stage separately."""
    raise NotImplementedError


def measure_io(reader, n_windows: int) -> dict:
    """Read bandwidth, to show whether scoring is limited by data or by compute."""
    raise NotImplementedError


def project(measurements: dict, n_slides: int, n_windows: int) -> dict:
    """Projected cost with and without caching, and the crossover in slides."""
    raise NotImplementedError


def main(argv: Optional[list] = None) -> int:
    """Run the measurements and print the projection."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
