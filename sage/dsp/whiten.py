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

import matplotlib.pyplot as plt


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
        corrupted_seconds: float = 2,
        device="cuda",
    ):
        super().__init__()

        self.device = device

        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.corrupted_len = int(round(corrupted_seconds * self.sample_rate))

        # Frequency resolution
        delta_f = sample_rate / seq_len
        self.delta_f = torch.tensor(delta_f).to(device=device)

        # Whitening
        whitening = 2 * self.delta_f / torch.sqrt(0.5 * fiducial_psds)
        # Final whitening moved to device
        whitening = whitening.to(device=device)

        # Register as buffer for compile friendliness
        self.register_buffer("whitening", whitening)  # (D, F)

    def remove_corrupted(self, x):
        # x_td_white or x: (B, D, T)
        T = x.shape[-1]
        start = self.corrupted_len
        end = T - self.corrupted_len
        return x[..., start:end]

    def whiten(self, X_fd: torch.Tensor) -> torch.Tensor:
        """
        X_fd: (B, D, F) complex64
        """

        # Apply whitening in FD
        X_white = X_fd * self.whitening.unsqueeze(0)

        # Back to time domain
        x_td_white = torch.fft.irfft(X_white, dim=-1, norm="forward") * self.delta_f

        # Remove corrupted regions
        # Typically we remove half the window length used
        # for estimating the PSD in Welch method
        x_td_white = self.remove_corrupted(x_td_white)

        return x_td_white
