#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : angular.py
Description     : Short description of the file

Created on 2026-02-16 10:45:51

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
from .uniform import Uniform


class UniformAngle(Uniform):
    TWO_PI = 2 * torch.pi

    def __init__(self):
        super().__init__(0.0, self.TWO_PI)

    @staticmethod
    def wrap(x):
        return torch.remainder(x, UniformAngle.TWO_PI)

    def sample(self, shape, device=None, dtype=torch.float32):
        # sample like uniform
        theta = super().sample(shape, device=device, dtype=dtype)

        # ensure strict periodic domain
        return self.wrap(theta)


class SinAngle(UniformAngle):
    PI = torch.pi

    def __init__(self, low=0.0, high=torch.pi):
        # domain [0, pi]
        self.low = float(low)
        self.high = float(high)

        self.cos_low = torch.cos(torch.tensor(self.high))
        self.cos_high = torch.cos(torch.tensor(self.low))

    def sample(self, shape, device=None, dtype=torch.float32):
        # uniform in cos(theta)
        u = torch.rand(shape, device=device, dtype=dtype)
        cos_theta = u * (self.cos_high - self.cos_low) + self.cos_low

        # inverse CDF
        theta = torch.arccos(cos_theta)

        return theta


class CosAngle(SinAngle):

    HALF_PI = torch.pi / 2

    def __init__(self, low=-torch.pi / 2, high=torch.pi / 2):
        self.low = float(low)
        self.high = float(high)

        self.sin_low = torch.sin(torch.tensor(self.low))
        self.sin_high = torch.sin(torch.tensor(self.high))

    def sample(self, shape, device=None, dtype=torch.float32):
        # uniform in sin(theta)
        u = torch.rand(shape, device=device, dtype=dtype)
        sin_theta = u * (self.sin_high - self.sin_low) + self.sin_low

        # inverse CDF
        theta = torch.arcsin(sin_theta)

        return theta


class UniformSolidAngle:

    def __init__(
        self,
        polar_name="theta",
        azimuthal_name="phi",
        polar_bounds=(0.0, torch.pi),
        azimuthal_bounds=(0.0, 2 * torch.pi),
    ):
        self.polar_name = polar_name
        self.azimuthal_name = azimuthal_name

        # reuse your fast samplers
        self.polar_sampler = SinAngle(*polar_bounds)
        self.azimuth_sampler = UniformAngle(*azimuthal_bounds)

    def sample(self, shape, device=None, dtype=torch.float32):
        theta = self.polar_sampler.sample(shape, device=device, dtype=dtype)
        phi = self.azimuth_sampler.sample(shape, device=device, dtype=dtype)

        return {
            self.polar_name: theta,
            self.azimuthal_name: phi,
        }
