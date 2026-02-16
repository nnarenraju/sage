#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : sky.py
Description     : Short description of the file

Created on 2026-02-16 10:53:06

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

# LOCAL
from .angular import UniformAngle, CosAngle


class UniformSky:
    """
    GPU-friendly uniform sky sampler.

    - Polar angle = declination `dec` in [-pi/2, pi/2]
    - Azimuthal angle = right ascension `ra` in [0, 2pi)
    """

    def __init__(
        self,
        polar_name="dec",
        azimuthal_name="ra",
        polar_bounds=(-torch.pi / 2, torch.pi / 2),
        azimuthal_bounds=(0.0, 2 * torch.pi),
    ):
        self.polar_name = polar_name
        self.azimuthal_name = azimuthal_name

        # Reuse your fast samplers
        self.polar_sampler = CosAngle(*polar_bounds)
        self.azimuth_sampler = UniformAngle(*azimuthal_bounds)

    def sample(self, shape, device=None, dtype=torch.float32):
        dec = self.polar_sampler.sample(shape, device=device, dtype=dtype)
        ra = self.azimuth_sampler.sample(shape, device=device, dtype=dtype)

        return {
            self.polar_name: dec,
            self.azimuthal_name: ra,
        }
