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
        device="cuda",
    ):
        super().__init__()

        self.device = device

        self.seq_len = seq_len
        self.sample_rate = sample_rate

        # Frequency resolution
        delta_f = sample_rate / seq_len
        delta_f = torch.tensor(delta_f).to(device=device)

        # Whitening factor:
        # sqrt(2 * delta_f) / sqrt(PSD)
        # (factor 2 because one-sided PSD)
        inv_sqrt_psd = 1.0 / torch.sqrt(fiducial_psds)
        whitening = torch.sqrt(2.0 * delta_f) * inv_sqrt_psd  # (D, F)
        whitening = whitening.to(device=device)

        # Register as buffer for compile friendliness
        self.register_buffer("whitening", whitening)  # (D, F)

    @staticmethod
    def remove_corrupted(self, timeseries):
        return timeseries[self.pad_len : timeseries.size()[-1] - self.pad_len]

    def whiten(self, X_fd: torch.Tensor) -> torch.Tensor:
        """
        X_fd: (B, D, F) complex64
        """

        # Apply whitening in FD
        X_white = X_fd * self.whitening.unsqueeze(0)

        print(self.whitening)

        plt.plot(self.whitening[0].detach().cpu().numpy())
        plt.show()

        # Back to time domain
        x_td_white = torch.fft.irfft(
            X_white,
            n=self.seq_len,
            dim=-1,
        )

        # Remove corrupted regions
        # Typically we remove half the window length used
        # for estimating the PSD in Welch method
        FiducialWhitening.remove_corrupted(x_td_white)

        plt.plot(x_td_white[0][0].detach().cpu().numpy())
        plt.show()

        return x_td_white
