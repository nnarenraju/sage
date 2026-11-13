#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Power-law priors on a bounded interval, with the radius uniform in a
3D volume.

Filename        : powerlaw.py
Description     : Short description of the file

Created on 2026-02-16 10:56:50

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

import torch


class UniformPowerLaw:
    """
    Power law with PDF proportional to ``r^(dim-1)`` on ``[low, high]``.

    ``dim=3`` is uniform in volume.

    Parameters
    ----------
    low : float
        Lower bound of the distribution.
    high : float
        Upper bound of the distribution.
    dim : int
        Dimensionality exponent; 3 gives uniform-in-volume.
    """

    name = "uniform_power_law"

    def __init__(self, low, high, dim):
        self.low = low
        self.high = high
        self.dim = dim

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Samples drawn by the inverse CDF, evaluated in float64.

        Parameters
        ----------
        shape : tuple of int
            Shape of the draw.
        device : str or torch.device or None
            Device of the draw.
        dtype : torch.dtype
            Dtype of the draw.
        generator : torch.Generator or None
            Generator of the draw.

        Returns
        -------
        torch.Tensor
            Samples of shape ``shape``, cast to ``dtype``.
        """
        u = torch.rand(shape, device=device, dtype=dtype, generator=generator).double()
        n = self.dim - 1
        return ((
            (self.high ** (n + 1) - self.low ** (n + 1)) * u + self.low ** (n + 1)
        ) ** (1.0 / (n + 1))).to(dtype)


class UniformRadius(UniformPowerLaw):
    """
    Radius uniform in a 3D volume, PDF proportional to ``r^2``.

    Parameters
    ----------
    low : float
        Lower bound of the radius.
    high : float
        Upper bound of the radius.
    """

    name = "uniform_radius"

    def __init__(self, low, high):
        super().__init__(low=low, high=high, dim=3)
