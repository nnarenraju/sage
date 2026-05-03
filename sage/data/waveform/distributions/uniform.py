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
    """
    Uniform distribution sampler over ``[low, high]``.

    GPU-friendly and generator-aware: passes the optional
    :class:`torch.Generator` through to :func:`torch.rand` for reproducible
    sampling.  Used by :class:`~sage.data.waveform.sampler.DistributionSampler`
    as the ``"uniform"`` prior type.

    Parameters
    ----------
    low : float
        Lower bound of the uniform distribution.
    high : float
        Upper bound of the uniform distribution.
    """

    def __init__(self, low, high):
        self.low = torch.tensor(low)
        self.scale = torch.tensor(high - low)

    def sample(self, shape, device=None, dtype=torch.float32, generator=None):
        """
        Draw samples from the uniform distribution.

        Parameters
        ----------
        shape : tuple[int, ...]
            Output shape.
        device : str or torch.device
            Target device.
        dtype : torch.dtype
            Output dtype.
        generator : torch.Generator or None
            Optional generator for reproducibility.

        Returns
        -------
        torch.Tensor
            Samples uniformly distributed in ``[low, high]``.
        """
        return self.low + self.scale * torch.rand(
            shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
