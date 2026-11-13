#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batched real-to-complex FFT of time-domain strain.

Filename        : fft.py
Description     : Short description of the file

Created on 2026-02-09 23:37:24

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


class BatchToFrequencyDomain:
    """
    Batched rFFT of real time-domain strain.

    Parameters
    ----------
    delta_t : float
        Sampling interval in seconds; stored, not used by the transform.
    """

    def __init__(self, *, delta_t: float):
        self.delta_t = delta_t

    def __call__(self, batch_td: torch.Tensor) -> torch.Tensor:
        """
        rFFT along the last axis with ``norm='forward'``.

        Parameters
        ----------
        batch_td : torch.Tensor, shape ``(B, D, T)``
            Real strain of ``D`` detectors, ``T`` samples each.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T//2 + 1)``
            Complex frequency-domain strain, scaled by ``1/T``.

        Raises
        ------
        ValueError
            If ``batch_td`` is not 3-dimensional.
        """
        if batch_td.ndim != 3:
            raise ValueError("Expected (B, D, T)")

        batch_fd = torch.fft.rfft(batch_td, dim=-1, norm='forward')
        return batch_fd
