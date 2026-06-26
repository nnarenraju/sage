#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : __init__.py
Description     : Short description of the file

Created on 2026-03-09 11:50:15

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

from .mscnn1d_att_resnet2d_cbam import (
    MSCNN1D_2DResNetCBAM,
    MSCNN1D_2DResNetCBAM_Heteroscedastic,
)
from .mscnn1d_att_resnet3d_cbam import MSCNN1Datt_3DResNetCBAM
from .mscnn1d_catt_resnet2d_cbam import MSCNN1D_catt_2DResNetCBAM
from .networks import MSCNN1D_2DResNetCBAM_HardMining
from .mc_dropout import enable_mc_dropout, mc_predict

__all__ = [
    "MSCNN1D_2DResNetCBAM",
    "MSCNN1Datt_3DResNetCBAM",
    "MSCNN1D_catt_2DResNetCBAM",
    "MSCNN1D_2DResNetCBAM_Heteroscedastic",
    "MSCNN1D_2DResNetCBAM_HardMining",
    "enable_mc_dropout",
    "mc_predict",
]

# Mamba3 models are optional: the mamba_ssm extension and attentive_mamba
# modules are gitignored / not synced on every machine. Import them only when
# present so the CNN models above stay usable without Mamba. Re-enables
# automatically once the mamba files are synced back in.
try:
    from .mamba_ssm import Mamba3
    from .attentive_mamba import BNSMamba3
    from .attentive_mamba_lite import BNSMamba3Lite
    from .attentive_mamba_tiny import BNSMamba3Tiny

    __all__ += ["Mamba3", "BNSMamba3", "BNSMamba3Lite", "BNSMamba3Tiny"]
except ImportError:
    pass
