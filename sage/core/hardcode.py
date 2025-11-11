#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : hardcode.py
Description     : All hardcoded parts of Sage

Created on 2025-11-08 10:19:45

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, Sage
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = University of Potsdam
__email__         = narenraju.nagarajan@uni-potsdam.de
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    This file is meant to be replaced with proper code whenever possible.
    If no algorithmic and safe way were found for a task, they are
    included here.

    Please expect everything here to be deprecated one day.

"""

# General
from typing import Union, List, Sequence

# LOCAL
from sage.core.utils import to_sequence


## None of the objects in this file are meant to be accessed by the user
## This is made clear with the use of '_' in the names

# --- Hardcoded on 08/11/2025 ---
# Detectors available obtained via https://gwosc.org/timeline
# GWOSC does not (yet) provide a way to query detector prefixes
# Using another package to retrieve prefixes is not safe
# Any det prefix outside this set will throw an error
# i.e. dets used *must* be a subset of the following
_DETECTORS = {"H1", "H2", "L1", "V1", "G1", "K1"}


def _check_detector_prefixes(dets: Union[str, Sequence[str]]):
    """Checking if detector prefixes are known

    Args:
        dets (_type_): _description_
    """
    dets = to_sequence(dets)
    assert all(
        [det in _DETECTORS for det in dets]
    ), "HARDCODE: Detector prefix not recognised!"
