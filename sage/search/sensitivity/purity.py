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
    """Expected signal count and purity from per-candidate p_astro."""
    raise NotImplementedError


def scaled_purity(
    p_astro: np.ndarray, vt_self: float, vt_reference: float, tier: str = "candidate"
) -> PurityResult:
    """
    Purity rescaled by a sensitivity ratio.

    Used when comparing a subthreshold count against a reference whose sensitivity
    differs, so the two are placed on the same footing.
    """
    raise NotImplementedError
