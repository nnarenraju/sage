#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : whiten.py
Description   : Short description of the file

Created on 2026-01-19 16:26:37

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import math
import numpy as np
import torch

# LOCAL
from sage.core.geometry import Window
from sage.core.pipeline import GWBatch, ProcessingState


class FiducialWhitening(torch.nn.Module):
    """
    Whiten frequency-domain strain using fixed, detector-specific fiducial ASDs.

    The whitening kernel is derived once from pre-computed fiducial ASDs and
    stored as a registered buffer so it moves to the correct device
    automatically and is included in ``torch.compile`` graphs.

    Pipeline (per sample)
    ---------------------
    1. Multiply FD strain by the whitening kernel:
       ``X_white = X_fd * whitening``  where
       ``whitening[d, f] = 2 Δf / (√0.5 · ASD[d, f])``.
    2. Convert back to time domain via inverse real FFT.
    3. Strip the corrupted edge samples introduced by the Welch
       estimation window (``padding_nsamples`` on each side).

    The ``@torch.no_grad()`` decorator on :meth:`forward` means this
    module **severs the autograd graph**.  Adversarial perturbations or any
    gradient-based optimisation must therefore operate on the *output* of
    this module, not on its FD input.

    Parameters
    ----------
    asds : torch.Tensor, shape ``(D, F)``
        Fiducial ASDs per detector (strain/sqrt(Hz)), on the padded grid.
    window : Window
        Window geometry; sets the frequency spacing and the padding removed.
    device : str or torch.device
        Device of the whitening kernel.

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

    def __init__(self, asds: torch.Tensor, window: Window, device):
        super().__init__()

        self.device = device

        self.seq_len = window.padded_length_in_samples
        self.sample_rate = window.sample_rate_in_hz
        self.corrupted_len = window.padding_in_samples

        # Frequency resolution
        delta_f = window.delta_f_in_hz
        self.delta_f = torch.tensor(delta_f).to(device=device)

        # Whitening
        whitening = 2 * self.delta_f / (math.sqrt(0.5) * asds)
        # Final whitening moved to device
        whitening = whitening.to(device=device)

        # Register as buffer for compile friendliness
        self.register_buffer("whitening", whitening)  # (D, F)

    def remove_corrupted(self, x):
        """
        Strip edge samples corrupted by the Welch estimation window.

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

    @staticmethod
    def next_state(state: ProcessingState) -> ProcessingState:
        """
        State after whitening.

        Parameters
        ----------
        state : ProcessingState
            State of the incoming batch.

        Returns
        -------
        ProcessingState
            Whitened TD_UNIFORM.

        Raises
        ------
        PipelineError
            If the batch is already whitened or not in the frequency domain.
        """
        return state.after_whiten().after_ifft()

    @torch.no_grad()
    def forward(self, batch: GWBatch) -> GWBatch:
        """
        Whiten frequency-domain strain: whiten, IFFT, strip the corrupted
        edges; returns ``TD_UNIFORM`` (real float32, shape ``(B, D, T_valid)``).

        Parameters
        ----------
        batch : GWBatch
            FD strain ``(B, D, F)`` complex.

        Returns
        -------
        GWBatch
            Whitened batch.
        """
        return self._forward_batch(batch)

    def _whiten_to_td(self, X_fd: torch.Tensor) -> torch.Tensor:
        """Whiten FD strain and convert to valid TD float32."""
        X_white = X_fd * self.whitening.unsqueeze(0)
        x_td    = torch.fft.irfft(X_white, dim=-1, norm="forward") * self.delta_f
        return self.remove_corrupted(x_td)

    def _forward_batch(self, batch: GWBatch) -> GWBatch:
        new_state = self.next_state(batch.state)
        return GWBatch(self._whiten_to_td(batch.data), new_state)
