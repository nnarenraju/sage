#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : purity.py
Description   : Expected astrophysical fraction of a candidate set.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

If the signal model is close to correct, summing p_astro over a candidate set gives the
expected number of real signals in it. Reporting this alongside a candidate list states
the expected contamination instead of leaving it to be inferred.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class PurityResult:
    """Expected signal count and purity of a candidate set."""

    n_candidates: int
    expected_signals: float
    expected_terrestrial: float
    purity: float
    tier: str

    def as_dict(self) -> dict:
        """Flat dict for tables and the manifest."""
        raise NotImplementedError


def purity(p_astro: np.ndarray, tier: str = "candidate") -> PurityResult:
    """
    Expected signal count and purity from per-candidate p_astro.

    sgwc-1's ``pastro.ipynb`` reports the count of candidates above a probability
    threshold; this is the same set summarised by its expectation instead of its count,
    which is what states the contamination rather than leaving it to be inferred.

    The array-level primitive. :func:`sage.search.candidates.expected_contamination` is
    the table-level entry point and reports the same two sums per tier -- the two must
    agree, and a test pins that, because a candidate list and its stated contamination
    disagreeing is exactly the kind of error a reader cannot catch.

    Purity is undefined for an empty set rather than one: nothing has been claimed, so
    there is no fraction of it that is real.
    """
    probability = np.asarray(p_astro, dtype=np.float64).ravel()
    if probability.size and (
        np.any(probability < 0.0) or np.any(probability > 1.0)
    ):
        raise ValueError(
            "p_astro must lie in [0, 1]; values outside it are not probabilities and "
            "their sum is not an expected count"
        )
    expected_signals = float(np.sum(probability))
    return PurityResult(
        n_candidates=int(probability.size),
        expected_signals=expected_signals,
        expected_terrestrial=float(np.sum(1.0 - probability)),
        purity=(
            expected_signals / probability.size if probability.size else float("nan")
        ),
        tier=str(tier),
    )


def scaled_purity(
    p_astro: np.ndarray, vt_self: float, vt_reference: float, tier: str = "candidate"
) -> PurityResult:
    """
    Purity rescaled by a sensitivity ratio.

    Used when comparing a subthreshold count against a reference whose sensitivity
    differs, so the two are placed on the same footing.

    .. note::

       Not implemented. The ratio it scales by is a ratio of sensitive volume-times, and
       the VT estimate is deferred -- Sage draws its own injections from the GWTC-3
       population rather than from one of the analytic distance distributions
       ``pycbc.sensitivity.volume_montecarlo`` assumes, so there is no estimator to take
       the ratio of yet. Left raising rather than approximated: a purity scaled by a
       guessed sensitivity ratio is a contamination statement that reads as measured.
    """
    raise NotImplementedError(
        "scaled_purity needs a sensitive volume-time ratio, and the VT estimate is "
        "deferred; use purity() for the unscaled expectation"
    )
