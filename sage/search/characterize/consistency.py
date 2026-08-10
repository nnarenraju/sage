#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : consistency.py
Description   : Tests that a candidate behaves like a signal rather than an artefact.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

A real signal is present in both detectors with a consistent arrival time, survives
changes to the analysis that should not matter, and leaves nothing structured behind
when a model of it is removed. An artefact typically fails at least one of these.

With only two detectors there is no combination in which a signal cancels, so that
particular test is unavailable and is reported as such rather than approximated.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConsistencyResult:
    """Outcome of one test."""

    name: str
    passed: Optional[bool]
    value: float
    detail: Dict[str, object]
    available: bool = True


def window_offset_stability(
    engine, strain, gps: float, offsets_s: Sequence[float], geometry
) -> ConsistencyResult:
    """
    Re-score the candidate with the analysis window shifted.

    A signal is scored consistently as the window moves, whereas a score driven by a
    short artefact falls away once the artefact leaves its preferred position.
    """
    raise NotImplementedError


def detector_ablation(engine, strain, gps: float, detectors: Sequence[str]) -> ConsistencyResult:
    """
    Re-score with one detector replaced by neighbouring noise.

    A coherent candidate loses most of its score; one driven by a single detector does not.
    """
    raise NotImplementedError


def arrival_time_coincidence(followup, detectors: Sequence[str]) -> ConsistencyResult:
    """Check the inter-detector arrival time against the light travel time."""
    raise NotImplementedError


def band_consistency(
    engine, strain, gps: float, bands: Sequence[Tuple[float, float]]
) -> ConsistencyResult:
    """Re-score in restricted frequency bands to see whether the evidence is broadband."""
    raise NotImplementedError


def coherent_versus_incoherent(strain, gps: float, detectors: Sequence[str]) -> ConsistencyResult:
    """
    Compare a coherent signal model against an independent-per-detector one.

    This is the most direct discriminator against instrumental artefacts, since faking a
    coherent excess requires a coincidence in both detectors.
    """
    raise NotImplementedError


def residual_test(strain, gps: float, template, detectors: Sequence[str]) -> ConsistencyResult:
    """Subtract the best-fit model and test the remainder for leftover structure."""
    raise NotImplementedError


def null_stream(strain, gps: float, detectors: Sequence[str]) -> ConsistencyResult:
    """
    Form the signal-cancelling combination, where the network allows one.

    Returns an unavailable result for a two-detector network.
    """
    raise NotImplementedError


def run_all(engine, strain, gps: float, detectors: Sequence[str], followup=None) -> Dict[str, ConsistencyResult]:
    """Run every applicable test."""
    raise NotImplementedError
