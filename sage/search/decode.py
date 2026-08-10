#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : decode.py
Description   : Decode the network's point-estimate head into physical quantities.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

The head emits a blocked layout ``[mu_0..mu_K, sraw_0..sraw_K]`` for the targets in
``cfg.do_point_estimate`` (production: tc, mchirp). These are point estimates with
heteroscedastic uncertainties, not parameter estimation; masses, spins, distance and
sky location are unavailable from the network.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass
class DecodedPE:
    """Physical point estimates and their standard deviations."""

    values: Dict[str, np.ndarray]
    sigmas: Dict[str, np.ndarray]
    at_prior_rail: Dict[str, np.ndarray]

    def column(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(value, sigma)`` for one target."""
        raise NotImplementedError


class PEDecoder:
    """
    Convert raw head outputs to physical units.

    Parameters
    ----------
    targets : sequence of str
        ``cfg.do_point_estimate`` ordering.
    param_sampler : object
        Provides the standardisation and min-max bounds used during training.
    """

    def __init__(self, targets: Sequence[str], param_sampler, pe_target_minmax: bool = False) -> None:
        raise NotImplementedError

    def split(self, point_estimates) -> Tuple["np.ndarray", "np.ndarray"]:
        """Split the blocked layout into means and raw sigmas."""
        raise NotImplementedError

    def sigma(self, raw_sigma) -> "np.ndarray":
        """Map raw sigma outputs to positive standard deviations."""
        raise NotImplementedError

    def decode(self, point_estimates) -> DecodedPE:
        """Un-standardise to physical values and flag prior-rail saturation."""
        raise NotImplementedError

    def tc_gps(self, window_start_gps: np.ndarray, tc_value: np.ndarray) -> np.ndarray:
        """Absolute coalescence time from a window start and the tc estimate."""
        raise NotImplementedError
