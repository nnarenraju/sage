#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : powerlaw.py
Description     : Short description of the file

Created on 2026-02-16 10:56:50

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


class UniformPowerLaw:
    """
    GPU-friendly power-law sampler for 1D parameters.

    Draws samples from the distribution whose PDF scales as
    ``r^(dim-1)`` over the interval ``[low, high]``.  For ``dim=3``
    this is uniform in volume (the standard astrophysical distance prior
    assuming a uniform spatial number density).

    Parameters
    ----------
    low : float
        Lower bound of the distribution.
    high : float
        Upper bound of the distribution.
    dim : int
        Dimensionality exponent; 3 gives uniform-in-volume (default).
    """

    name = "uniform_power_law"

    def __init__(self, low, high, dim=3):
        self.low = low
        self.high = high
        self.dim = dim

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """Sample a batch from the power-law distribution on GPU."""
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        n = self.dim - 1
        return (
            (self.high ** (n + 1) - self.low ** (n + 1)) * u + self.low ** (n + 1)
        ) ** (1.0 / (n + 1))


class UniformRadius(UniformPowerLaw):
    """Uniform in volume (3D sphere) radius sampler."""

    name = "uniform_radius"

    def __init__(self, low=0.0, high=1.0):
        super().__init__(low=low, high=high, dim=3)
