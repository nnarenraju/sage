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
    Whiten frequency-domain strain using fixed, detector-specific fiducial PSDs.

    The whitening kernel is derived once from pre-computed fiducial ASDs and
    stored as a registered buffer so it moves to the correct device
    automatically and is included in ``torch.compile`` graphs.

    Pipeline (per sample)
    ---------------------
    1. Multiply FD strain by the whitening kernel:
       ``X_white = X_fd * whitening``  where
       ``whitening[d, f] = 2 Δf / (√0.5 · ASD[d, f])``.
    2. Convert back to time domain via inverse real FFT.
    3. Strip the corrupted edge samples introduced by the Welch PSD
       estimation window (``padding_nsamples`` on each side).

    The ``@torch.no_grad()`` decorator on :meth:`forward` means this
    module **severs the autograd graph**.  Adversarial perturbations or any
    gradient-based optimisation must therefore operate on the *output* of
    this module, not on its FD input.

    Parameters
    ----------
    **kwargs
        Forwarded to ``nn.Module.__init__``.

    Attributes
    ----------
    whitening : torch.Tensor, shape ``(D, F)``
        Per-detector, per-frequency whitening kernel (registered buffer).
    corrupted_len : int
        Number of samples removed from each end of the whitened time series.

    Input / Output
    --------------
    forward(X_fd) : (B, D, F) complex64 → (B, D, T_valid) float32
        where ``T_valid = seq_len - 2 * corrupted_len``.
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
        """
        Strip edge samples corrupted by the Welch PSD estimation window.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``
            Whitened time-domain strain (full length, including corrupted ends).

        Returns
        -------
        torch.Tensor, shape ``(B, D, T - 2 * corrupted_len)``
            Valid central samples only.
        """
        # x_td_white or x: (B, D, T)
        T = x.shape[-1]
        start = self.corrupted_len
        end = T - self.corrupted_len
        return x[..., start:end]

    @torch.no_grad()
    def forward(self, X_fd: torch.Tensor) -> torch.Tensor:
        """
        Whiten and convert FD strain to a valid TD time series.

        .. note::
            This method runs under ``@torch.no_grad()``, severing the
            autograd graph.  Gradient-based methods (adversarial noise,
            saliency maps) must operate on the *output* tensor.

        Parameters
        ----------
        X_fd : torch.Tensor, shape ``(B, D, F)`` complex64
            Frequency-domain strain for a batch of B windows across D detectors.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T_valid)`` float32
            Whitened time-domain strain with corrupted edges removed.
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
