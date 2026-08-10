#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : tables.py
Description   : Publication tables.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Tables are generated rather than transcribed, so a change upstream cannot leave a stale
number in the manuscript.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence


def candidate_table(candidates, path: str | Path, tier: int = 1, level: float = 0.9) -> Path:
    """
    Per-candidate table: name, time, significance, probability and recovered parameters.

    Quantities that require parameter estimation are omitted for candidates that did not
    receive it, rather than filled from the search's point estimates.
    """
    raise NotImplementedError


def comparison_table(comparison, path: str | Path) -> Path:
    """
    Event-by-catalogue comparison.

    Entries unique to one catalogue are marked, and events outside a catalogue's searched
    region are distinguished from ones it searched without finding.
    """
    raise NotImplementedError


def sensitivity_table(fiducial_points, path: str | Path) -> Path:
    """Sensitivity at the reference masses, with uncertainties and coverage notes."""
    raise NotImplementedError


def configuration_table(spec, manifest, path: str | Path) -> Path:
    """
    The analysis configuration, as a methods-section table.

    Covers the data used, conditioning, ranking statistic, background construction,
    thresholds and livetimes.
    """
    raise NotImplementedError


def livetime_table(manifest, path: str | Path) -> Path:
    """Analysed and background livetime per observing run."""
    raise NotImplementedError
