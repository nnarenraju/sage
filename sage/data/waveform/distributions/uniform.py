#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Uniform prior on a bounded interval.

Filename        : uniform.py
Description     : Short description of the file

Created on 2026-02-16 10:38:20

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


class Uniform:
    """
    Uniform distribution on ``[low, high)``.

    Parameters
    ----------
    low : float
        Lower bound of the uniform distribution.
    high : float
        Upper bound of the uniform distribution.
    """

    def __init__(self, low, high):
        # Python floats, so a float64 draw uses the exact bounds
        self.low = float(low)
        self.scale = float(high) - float(low)

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Samples drawn uniformly from ``[low, high)``.

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
            Samples of shape ``shape``.
        """
        return self.low + self.scale * torch.rand(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
