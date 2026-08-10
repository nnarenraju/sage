#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : compare.py
Description   : Sensitivity comparison against the reference pipelines.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The published injection sets carry each reference pipeline's own significance columns,
so their sensitivity can be recomputed here with the identical estimator, population and
threshold. That removes the estimator itself as a source of difference and doubles as a
check on this implementation: recomputing a pipeline's published value should reproduce
it.

Comparability has one limit that must be carried into the figure. Sage analyses only
times when both detectors are observing, whereas the reference pipelines also admit
single-detector and larger-network candidates. Either the comparison is restricted to
the same coincidence type, or the axis says plainly that it is not.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from sage.search.sensitivity.vt import VTResult


@dataclass
class PipelineComparison:
    """Sensitivity of each pipeline under one estimator and population."""

    results: Dict[str, VTResult]
    population: str
    far_threshold_per_yr: float
    coincidence_restricted: bool
    note: str = ""

    def ratio_to(self, reference: str) -> Dict[str, float]:
        """Sensitivity of each pipeline relative to one reference."""
        raise NotImplementedError


def compare_pipelines(
    injections,
    sage_match,
    population,
    analysis_time_s: float,
    pipelines: Optional[Sequence[str]] = None,
    far_threshold_per_yr: float = 1.0,
    restrict_coincidence: bool = True,
) -> PipelineComparison:
    """Recompute every pipeline's sensitivity with the same estimator and settings."""
    raise NotImplementedError


def validate_against_published(
    injections, pipeline: str, published_vt: float, rtol: float = 0.05
) -> Dict[str, float]:
    """Reproduce a pipeline's published sensitivity as a check on this estimator."""
    raise NotImplementedError


def coincidence_mask(injections, pipeline: str, detectors: Sequence[str]) -> np.ndarray:
    """Injections a pipeline recovered in the same coincidence type Sage analyses."""
    raise NotImplementedError
