#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : fft.py
Description     : Short description of the file

Created on 2026-02-09 23:37:24

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


class BatchToFrequencyDomain:
    def __init__(self, *, delta_t: float):
        self.delta_t = delta_t

    def __call__(self, batch_td: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch_td: (B, D, T) real
        Returns:
            batch_fd: (B, D, F) complex
        """
        if batch_td.ndim != 3:
            raise ValueError("Expected (B, D, T)")

        # rFFT over time dimension
        batch_fd = torch.fft.rfft(batch_td, dim=-1)
        return batch_fd
