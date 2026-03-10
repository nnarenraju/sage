#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : snr_rescaling.py
Description     : Short description of the file

Created on 2026-03-10 03:59:41

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
import torch.nn as nn


class HalfNorm(nn.Module):

    def __init__(self, scale=1.0, loc=0.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("loc", torch.tensor(loc))

    def forward(self, batch_size: int):
        x = torch.randn(batch_size, device=self.scale.device).abs()
        return x * self.scale + self.loc
