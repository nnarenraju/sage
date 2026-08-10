#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : characterize.py
Description   : Characterise candidates from a completed search.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Run after a search, against its candidate list. Kept separate from the search because it
is per-event work, it needs the parameter-estimation environment, and it is normally
applied to a chosen few candidates rather than to all of them.

Usage
-----
    # screen everything above the confident threshold
    python characterize.py --campaign o4a_HL --tier 1

    # one candidate, in full
    python characterize.py --campaign o4a_HL --event SGW230814_230901 --level full

    # add parameter estimation, submitted into its own environment
    python characterize.py --campaign o4a_HL --event SGW230814_230901 --pe

Results are written back into the campaign store, and the candidate tiers are re-derived
so that anything the vetting rejects is demoted.
"""

import argparse
from typing import Optional


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments."""
    raise NotImplementedError


def main(argv: Optional[list] = None) -> int:
    """Characterise the selected candidates and update the store."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
