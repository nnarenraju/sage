#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whitening of frequency-domain strain by fixed fiducial ASDs, returning
the valid time-domain samples.

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

import math
import numpy as np
import torch

from sage.core.geometry import Window
from sage.core.pipeline import GWBatch, ProcessingState


class FiducialWhitening(torch.nn.Module):
    """
    Whiten frequency-domain strain by fixed per-detector fiducial ASDs.

    Multiplies by ``2 delta_f / (sqrt(0.5) ASD)``, applies the inverse
    rFFT times ``delta_f``, and strips ``corrupted_len`` samples from
    each end. :meth:`forward` runs under ``no_grad``.

    Parameters
    ----------
    asds : torch.Tensor, shape ``(D, F)``
        Fiducial ASD per detector (strain/sqrt(Hz)), padded grid.
    window : Window
        Window geometry; sets the frequency spacing and the padding.
    device : str or torch.device
        Device of the whitening kernel.

    Attributes
    ----------
    whitening : torch.Tensor, shape ``(D, F)``
        Whitening kernel per detector and frequency, a buffer.
    corrupted_len : int
        Samples removed from each end of the whitened time series.
    """

    def __init__(self, asds: torch.Tensor, window: Window, device):
        super().__init__()

        self.device = device

        self.seq_len = window.padded_length_in_samples
        self.sample_rate = window.sample_rate_in_hz
        self.corrupted_len = window.padding_in_samples

        delta_f = window.delta_f_in_hz
        self.delta_f = torch.tensor(delta_f).to(device=device)

        whitening = 2 * self.delta_f / (math.sqrt(0.5) * asds)
        whitening = whitening.to(device=device)

        self.register_buffer("whitening", whitening)  # (D, F)

    def remove_corrupted(self, x):
        """
        Strip ``corrupted_len`` samples from each end.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``
            Whitened time-domain strain, full length.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T - 2 * corrupted_len)``
            Valid central samples.
        """
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
            If the batch is already whitened or not in the frequency
            domain.
        """
        return state.after_whiten().after_ifft()

    @torch.no_grad()
    def forward(self, batch: GWBatch) -> GWBatch:
        """
        Whiten a frequency-domain batch to valid time-domain samples.

        Parameters
        ----------
        batch : GWBatch
            FD strain ``(B, D, F)``, complex.

        Returns
        -------
        GWBatch
            Whitened ``TD_UNIFORM`` strain ``(B, D, T_valid)``, with
            ``T_valid = T - 2 * corrupted_len``.

        Raises
        ------
        PipelineError
            If the batch is already whitened or not in the frequency
            domain.
        """
        return self._forward_batch(batch)

    def _whiten_to_td(self, X_fd: torch.Tensor) -> torch.Tensor:
        """
        Whiten FD strain and return its valid time-domain samples.

        Parameters
        ----------
        X_fd : torch.Tensor, shape ``(B, D, F)``
            Complex FD strain.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T_valid)``
            Whitened real strain.
        """
        X_white = X_fd * self.whitening.unsqueeze(0)
        x_td    = torch.fft.irfft(X_white, dim=-1, norm="forward") * self.delta_f
        return self.remove_corrupted(x_td)

    def _forward_batch(self, batch: GWBatch) -> GWBatch:
        """
        Whitened batch with its new processing state.

        Parameters
        ----------
        batch : GWBatch
            FD strain ``(B, D, F)``, complex.

        Returns
        -------
        GWBatch
            Whitened ``TD_UNIFORM`` strain.

        Raises
        ------
        PipelineError
            If the batch is already whitened or not in the frequency
            domain.
        """
        new_state = self.next_state(batch.state)
        return GWBatch(self._whiten_to_td(batch.data), new_state)
