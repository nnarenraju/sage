#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Noise samplers: real detector noise, synthetic coloured noise,
recolouring, glitch oversampling and mined low-FAR start times.

Filename        : __init__.py
Description     : Short description of the file

Created on 2026-01-19 23:28:02

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
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
from .testing_noise import TestNoiseSampler

from .white_noise import (
    sample_synthetic_noise,
    available_asds,
    WhiteGaussianNoiseSampler,
)
from .recolour import RecolourPostprocess
from .glitch_sampler import GlitchOversampledNoiseSampler
from .lowfar_noise import StartTimeDataset

# The miner in sage.data.noise.cma_mae_mining needs pyribs; it is not
# re-exported so that importing this package does not require pyribs.

__all__ = [
    "HDF5SingleNoiseSampler",
    "MemmapSingleNoiseSampler",
    "MemmapNoiseSampler",
    "TestNoiseSampler",
    "sample_synthetic_noise",
    "available_asds",
    "WhiteGaussianNoiseSampler",
    "RecolourPostprocess",
    "GlitchOversampledNoiseSampler",
    "StartTimeDataset",
]
