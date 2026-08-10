#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : build_sensitivity.py
Description   : Figure data for sensitivity, injections and pipeline comparison.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Sensitivity values are read from the products the sensitivity stage wrote. Recomputing
them here would create a second set of numbers that could drift from the analysis and
from the released data.
"""

from pathlib import Path
from typing import Dict, Optional

from sage.search.figdata.product import FigData


def vt_versus_far(spec) -> FigData:
    """Sensitive volume-time against the false-alarm-rate threshold."""
    raise NotImplementedError


def vt_versus_parameter(spec) -> FigData:
    """Sensitivity binned in total mass, chirp mass, mass ratio and redshift."""
    raise NotImplementedError


def pipeline_comparison(spec) -> FigData:
    """
    Sensitivity against the reference pipelines on the same injections.

    Records whether the comparison was restricted to a common coincidence type, so the
    figure can label the axis honestly when it was not.
    """
    raise NotImplementedError


def injection_recovery(spec) -> FigData:
    """Found and missed injections, and efficiency in false-alarm-rate units."""
    raise NotImplementedError


def sensitive_distance(spec) -> FigData:
    """Sensitive distance against mass."""
    raise NotImplementedError


def range_over_time(spec) -> FigData:
    """Detector range across the run, with its distribution."""
    raise NotImplementedError


def pastro_reliability(spec) -> FigData:
    """
    Predicted against realised astrophysical fraction, from injections.

    This is the check that the reported probability means what it claims.
    """
    raise NotImplementedError


def build(spec, figures: Optional[list] = None) -> Dict[str, Path]:
    """Build every sensitivity figure data product."""
    raise NotImplementedError
