#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : run.py
Description   : Stage driver for p_astro.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress
"""

from pathlib import Path
from typing import Optional

from sage.search.pastro.validate import ValidationReport


def run(spec, resume: bool = True) -> ValidationReport:
    """
    Fit the mixture for one observing run and assign per-trigger probabilities.

    Runs in order: build the shared support from the analysis threshold, estimate both
    densities on it, gate on the likelihood-ratio ordering, infer and marginalise the
    rates, assign probabilities, then run the validation suite. A failed gate stops the
    stage without writing products.
    """
    raise NotImplementedError


def main(argv: Optional[list] = None) -> int:
    """Command-line entry point."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
