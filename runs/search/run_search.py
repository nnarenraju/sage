#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : run_search.py
Description   : Search an observing run with a trained network.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

One call from trained weights to candidates, sensitivity and figures.

Usage
-----
    python run_search.py \
        --checkpoint /work/nagarajan/sage_runs/o4a/CHECKPOINTS/best.pt \
        --config runs.o4a.config_HL \
        --run O4a

    # see the plan and the projected cost without running anything
    python run_search.py ... --dry-run

    # shallow background first, to exercise every step end to end
    python run_search.py ... --n-slides 8

    # stop once the candidate list exists
    python run_search.py ... --stop-after candidates

Per-event characterization and parameter estimation are not run here; use
``characterize.py`` against the candidate list this produces.
"""

import argparse
from typing import Optional


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments mirroring :func:`sage.search.pipeline.run_search`."""
    raise NotImplementedError


def main(argv: Optional[list] = None) -> int:
    """Run the search and print the summary."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
