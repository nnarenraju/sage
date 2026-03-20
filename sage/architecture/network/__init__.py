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

from .mscnn1d_att_resnet2d_cbam import MSCNN1D_2DResNetCBAM
from .mscnn1d_att_resnet3d_cbam import MSCNN1Datt_3DResNetCBAM
from .mscnn1d_catt_resnet2d_cbam import MSCNN1D_catt_2DResNetCBAM

__all__ = [
    "MSCNN1D_2DResNetCBAM",
    "MSCNN1Datt_3DResNetCBAM",
    "MSCNN1D_catt_2DResNetCBAM",
]
