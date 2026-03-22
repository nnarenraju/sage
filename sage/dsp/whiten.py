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
import math
import torch
import matplotlib.pyplot as plt

# LOCAL
from sage.data.psd import get_fiducial_psds
from sage.core.config import get_cfg, get_data_cfg


class FiducialWhitening(torch.nn.Module):
    """
    Whiten frequency-domain strain using fixed fiducial PSDs.

    Input:
        X_fd: (B, D, F) complex64

    Output:
        x_td_white: (B, D, T) float32
    """

    GRAPH_READY = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setup configs
        cfg = get_cfg()
        data_cfg = get_data_cfg()

        # Get fiducial psds
        fiducial_psds = get_fiducial_psds()

        self.device = cfg.device

        self.seq_len = data_cfg.padded_length_in_nsamples
        self.sample_rate = data_cfg.sample_rate
        self.corrupted_len = data_cfg.padding_nsamples

        # Frequency resolution
        delta_f = data_cfg.sample_rate / self.seq_len
        self.delta_f = torch.tensor(delta_f).to(device=cfg.device)

        # Whitening
        whitening = 2 * self.delta_f / (math.sqrt(0.5) * fiducial_psds)
        # Final whitening moved to device
        whitening = whitening.to(device=cfg.device)

        # Register as buffer for compile friendliness
        self.register_buffer("whitening", whitening)  # (D, F)

    def remove_corrupted(self, x):
        # x_td_white or x: (B, D, T)
        T = x.shape[-1]
        start = self.corrupted_len
        end = T - self.corrupted_len
        return x[..., start:end]

    @torch.no_grad()
    def forward(self, X_fd: torch.Tensor) -> torch.Tensor:
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
