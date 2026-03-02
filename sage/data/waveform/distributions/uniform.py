#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : uniform.py
Description     : Short description of the file

Created on 2026-02-16 10:38:20

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

# Packages
import torch


class Uniform:
    def __init__(self, low, high):
        self.low = torch.tensor(low)
        self.scale = torch.tensor(high - low)

    def sample(self, shape, device=None, dtype=torch.float32):
        return self.low + self.scale * torch.rand(
            shape,
            device=device,
            dtype=dtype,
        )
