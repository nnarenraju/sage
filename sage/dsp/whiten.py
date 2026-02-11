#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : whiten.py
Description   : Short description of the file

Created on 2026-01-19 16:26:37

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import torch


class FiducialWhitening(torch.nn.Module):
    """
    Whiten frequency-domain strain using fixed fiducial PSDs.

    Input:
        X_fd: (B, D, F) complex64

    Output:
        x_td_white: (B, D, T) float32
    """

    def __init__(
        self,
        fiducial_psds: torch.Tensor,  # (D, F) float32
        seq_len: int,
        sample_rate: float,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.eps = eps

        # Frequency resolution
        delta_f = sample_rate / seq_len

        # Whitening factor:
        # sqrt(2 * delta_f) / sqrt(PSD)
        # (factor 2 because one-sided PSD)
        whitening = torch.sqrt(2.0 * delta_f / (fiducial_psds + eps))  # (D, F)

        # Register as buffer for compile friendliness
        self.register_buffer("whitening", whitening)  # (D, F)

    def forward(self, X_fd: torch.Tensor) -> torch.Tensor:
        """
        X_fd: (B, D, F) complex64
        """

        # Apply whitening in FD
        X_white = X_fd * self.whitening.unsqueeze(0)

        # Back to time domain
        x_td_white = torch.fft.irfft(
            X_white,
            n=self.seq_len,
            dim=-1,
        )

        return x_td_white
