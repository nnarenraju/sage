#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Packages
import torch


class BatchToFrequencyDomain:
    """
    Callable that converts a batch of real time-domain strain to the
    frequency domain via a real-to-complex FFT (``torch.fft.rfft``).

    Parameters
    ----------
    delta_t : float
        Sampling interval in seconds (= 1 / sample_rate).  Stored for
        reference; not currently used in the computation but available for
        downstream normalisation.
    """

    def __init__(self, *, delta_t: float):
        self.delta_t = delta_t

    def __call__(self, batch_td: torch.Tensor) -> torch.Tensor:
        """
        Transform a batch of time-domain signals to frequency domain.

        Parameters
        ----------
        batch_td : torch.Tensor, shape ``(B, D, T)``
            Batch of real-valued time-domain strain windows; ``D`` detectors,
            ``T`` samples.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T//2 + 1)``
            Complex frequency-domain strain.

        Raises
        ------
        ValueError
            If ``batch_td`` is not 3-dimensional.
        """
        if batch_td.ndim != 3:
            raise ValueError("Expected (B, D, T)")

        # rFFT over time dimension
        batch_fd = torch.fft.rfft(batch_td, dim=-1, norm='forward')
        return batch_fd
