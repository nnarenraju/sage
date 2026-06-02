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
from .hard_mining import HardSampleBuffer, HardSampleMiner
from .glitch_sampler import GlitchOversampledNoiseSampler
from .lowfar_noise import (
    StartTimeDataset,
    StartTimeNoiseSampler,
    BruteForceMiner,
    MAPElitesMiner,
    CEMRareEventMiner,
)
# NOTE: qd_mining is gitignored and not present on this machine. Commented out
# until it is synced here (the QD hard-mining symbols are not used by the PSD
# or training pipelines). Re-enable once sage/data/noise/qd_mining.py exists.
# from .qd_mining import (
#     NoiseSVDProjector,
#     SharedHardNoiseBank,
#     CMAMEMiner,
#     CMAMEGAMiner,
#     make_miner_preprocessor,
# )

__all__ = [
    "HDF5SingleNoiseSampler",
    "MemmapSingleNoiseSampler",
    "MemmapNoiseSampler",
    "WhiteGaussianNoiseSampler",
    "RecolourPostprocess",
    "HardSampleBuffer",
    "HardSampleMiner",
    "GlitchOversampledNoiseSampler",
    "StartTimeDataset",
    "StartTimeNoiseSampler",
    "BruteForceMiner",
    "MAPElitesMiner",
    "CEMRareEventMiner",
    # qd_mining symbols (commented out until qd_mining.py is synced here):
    # "NoiseSVDProjector",
    # "SharedHardNoiseBank",
    # "CMAMEMiner",
    # "CMAMEGAMiner",
    # "make_miner_preprocessor",
]
