#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : __init__.py
Description     : Short description of the file

Created on 2026-01-19 23:28:02

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

from .real_noise import (
    HDF5SingleNoiseSampler,
    MemmapSingleNoiseSampler,
    MemmapNoiseSampler,
)

from .white_noise import WhiteGaussianNoiseSampler
from .recolour import RecolourPostprocess
from .glitch_sampler import GlitchOversampledNoiseSampler
from .lowfar_noise import StartTimeDataset

# NOTE: the hard-negative miner lives in ``sage.data.noise.cma_mae_mining`` and
# is imported directly by the hard-mining trainer (it pulls in pyribs).  It is
# deliberately NOT re-exported here so that importing this package — and the
# default training path — never requires pyribs.

__all__ = [
    "HDF5SingleNoiseSampler",
    "MemmapSingleNoiseSampler",
    "MemmapNoiseSampler",
    "WhiteGaussianNoiseSampler",
    "RecolourPostprocess",
    "GlitchOversampledNoiseSampler",
    "StartTimeDataset",
]
