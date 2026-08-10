#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : emission.py
Description   : Per-injection output in the shared sensitivity-estimate schema.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Emitting the same per-injection fields as the reference pipelines is what allows an
external group to recompute Sage's sensitivity and compare it directly. Injection-side
quantities are copied from the release rather than recomputed.

Sage's network estimates only coalescence time and chirp mass, so mass and spin columns
carry a provenance of parameter estimation or are left undefined; a follow-up template's
parameters are not parameter estimates and are never written as though they were.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

EMISSION_COLUMNS = (
    "sage_far",
    "sage_p_astro",
    "sage_ranking_statistic",
    "sage_gps_time",
    "sage_rec_mchirp",
    "sage_dt",
)

MISSED_DEFAULTS = {
    "sage_far": np.inf,
    "sage_p_astro": 0.0,
    "sage_ranking_statistic": np.nan,
}


def emit(
    injections,
    match,
    path: str | Path,
    observing_run: str,
    total_analysis_time_s: float,
    attrs: Optional[Dict[str, object]] = None,
) -> Path:
    """Write the per-injection recovery file for one observing run."""
    raise NotImplementedError


def validate_emission(path: str | Path) -> Dict[str, object]:
    """Check required fields, missed-injection defaults and the recorded accounting."""
    raise NotImplementedError
