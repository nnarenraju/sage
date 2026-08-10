#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : fiducial.py
Description   : Sensitivity quoted at reference component masses.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Reference points let sensitivity be compared across pipelines without agreeing on a
population. Points whose mass support lies outside the range the network was trained on
are reported as out of range rather than given a number, since a reweighted value there
measures extrapolation and would be read as sensitivity.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from sage.search.sensitivity.vt import VTResult


@dataclass
class FiducialPoint:
    """One reference mass pair and its sensitivity."""

    m1: float
    m2: float
    log_width: float
    result: Optional[VTResult]
    in_training_range: bool
    note: str = ""


def fiducial_points(
    injections,
    match,
    analysis_time_s: float,
    masses: Sequence[float],
    training_box: Tuple[float, float] = (7.0, 50.0),
    log_width: float = 0.1,
    far_threshold_per_yr: float = 1.0,
) -> Sequence[FiducialPoint]:
    """Evaluate sensitivity at each reference mass, flagging out-of-range points."""
    raise NotImplementedError


def in_training_range(
    m1: float, m2: float, log_width: float, box: Tuple[float, float], credible: float = 0.9
) -> bool:
    """Whether the reference kernel's credible mass range lies inside the trained box."""
    raise NotImplementedError
