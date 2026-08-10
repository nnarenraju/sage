#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : gmst.py
Description   : GPS to Greenwich mean sidereal time.

Created on 2026-08-09

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = inProgress

Antenna response depends on sidereal time, so injections must be spread over the
sidereal day. Placing every injection at one fixed epoch leaves the sky response
un-marginalised and biases recovered sensitivity.
"""

from typing import Union

import numpy as np


def gps_to_gmst_rad(gps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Greenwich mean sidereal angle in radians."""
    raise NotImplementedError


def cross_check_gmst(gps: np.ndarray, atol_rad: float = 1e-9) -> float:
    """Compare the primary implementation against an independent one."""
    raise NotImplementedError


def sidereal_uniformity(gps: np.ndarray) -> dict:
    """Test whether injection times are uniform over the sidereal day."""
    raise NotImplementedError
