#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : hardcode.py
Description     : All hardcoded parts of Sage

Created on 2025-11-08 10:19:45

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, Sage
__license__       = GPL-3.0-or-later
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
# Storing as a set is immutable and fine for torch
_DETECTORS = {"H1", "H2", "L1", "V1", "G1", "K1"}


def _check_detector_prefixes(dets: Union[str, Sequence[str]]):
    """Checking if detector prefixes are known

    Args:
        dets (_type_): _description_
    """
    dets = to_sequence(dets)
    assert all(
        [det in _DETECTORS for det in dets]
    ), "HARDCODE: Detector prefix(es) not recognised!"


# --- Hardcoded on 25/01/2026 ---
# Detector response computation requires DET metadata from LAL
# The following data has been hardcoded from LAL for each detector
# longitude, latitude, yangle, xangle, xaltitude, yaltitude
#
# You can obtain the above information from LAL using:
# import pycbc
# pref = 'H1' --> Change to other det names from _DETECTORS
# lalsim = pycbc.libutils.import_optional('lalsimulation')
# lal_det = lalsim.DetectorPrefixToLALDetector(pref).frDetector
#
# print([pref, # name
#     lal_det.vertexLongitudeRadians, # longitude
#     lal_det.vertexLatitudeRadians, # latitude
#     lal_det.vertexElevation, # height
#     lal_det.xArmAzimuthRadians, # xangle
#     lal_det.yArmAzimuthRadians, # yangle
#     lal_det.xArmMidpoint * 2, # xlength
#     lal_det.yArmMidpoint * 2, # ylength
#     lal_det.xArmAltitudeRadians, # xaltitude
#     lal_det.yArmAltitudeRadians,] # yaltitude
# )
#
_DETMETADATA = {
    "H1": {
        "longitude": -2.08405676917,
        "latitude": 0.81079526383,
        "height": 142.5540008544922,
        "xangle": 5.654877185821533,
        "yangle": 4.084080696105957,
        "xlength": 3995.083984375,
        "ylength": 3995.0439453125,
        "xaltitude": -0.0006195000023581088,
        "yaltitude": 1.249999968422344e-05,
    },
    "H2": {
        "longitude": -2.08405676917,
        "latitude": 0.81079526383,
        "height": 142.5540008544922,
        "xangle": 5.654877185821533,
        "yangle": 4.084080696105957,
        "xlength": 2009.0,
        "ylength": 2009.0,
        "xaltitude": -0.0006195000023581088,
        "yaltitude": 1.249999968422344e-05,
    },
    "L1": {
        "longitude": -1.58430937078,
        "latitude": 0.53342313506,
        "height": -6.573999881744385,
        "xangle": 4.403177738189697,
        "yangle": 2.8323814868927,
        "xlength": 3995.14990234375,
        "ylength": 3995.14990234375,
        "xaltitude": -0.00031209998996928334,
        "yaltitude": -0.000610699993558228,
    },
    "V1": {
        "longitude": 0.18333805213,
        "latitude": 0.76151183984,
        "height": 51.88399887084961,
        "xangle": 0.3391628563404083,
        "yangle": 5.051551818847656,
        "xlength": 3000.0,
        "ylength": 3000.0,
        "xaltitude": 0.0,
        "yaltitude": 0.0,
    },
    "G1": {
        "longitude": 0.17116780435,
        "latitude": 0.91184982752,
        "height": 114.42500305175781,
        "xangle": 1.1936010122299194,
        "yangle": 5.830392837524414,
        "xlength": 600.0,
        "ylength": 600.0,
        "xaltitude": 0.0,
        "yaltitude": 0.0,
    },
    "K1": {
        "longitude": 2.396441015,
        "latitude": 0.6355068497,
        "height": 414.1809997558594,
        "xangle": 1.0541130304336548,
        "yangle": -0.5166798233985901,
        "xlength": 3026.507080078125,
        "ylength": 3023.221923828125,
        "xaltitude": 0.0031413999386131763,
        "yaltitude": -0.0036269999109208584,
    },
}
