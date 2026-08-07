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
from sage.data.asd import get_fiducial_asds
from sage.core.config import get_cfg, get_data_cfg
from sage.core.pipeline import GWBatch, Grid, ProcessingState


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Setup configs
        cfg = get_cfg()
        data_cfg = get_data_cfg()

        # Per-detector fiducial ASDs (strain/sqrt(Hz)); see sage.data.asd
        fiducial_asds = get_fiducial_asds()

        self.device = cfg.device

        self.seq_len = data_cfg.padded_length_in_nsamples
        self.sample_rate = data_cfg.sample_rate
        self.corrupted_len = data_cfg.padding_nsamples

        # Frequency resolution
        delta_f = data_cfg.sample_rate / self.seq_len
        self.delta_f = torch.tensor(delta_f).to(device=cfg.device)

        # Whitening
        whitening = 2 * self.delta_f / (math.sqrt(0.5) * fiducial_asds)
        # Final whitening moved to device
        whitening = whitening.to(device=cfg.device)

        # Register as buffer for compile friendliness
        self.register_buffer("whitening", whitening)  # (D, F)

        # ~Unit-variance normalisation for the FD_COARSE (stay-in-FD) path.
        # The TD path gets unit variance from the ``irfft(norm="forward")·Δf`` step
        # in ``_whiten_to_td``; the FD_COARSE branch skips it, leaving the whitened
        # data sub-unity.  We restore ~O(1) with a SINGLE analytic constant
        # (dividing the 2Δf/(√0.5·ASD) whitening by 2·Δf^1.5, i.e. the proper
        # √2/(ASD·√Δf) form).  It is a shared scalar -- deliberately NOT a
        # per-detector or data-driven fit: each detector is already divided by its
        # own ASD, so a common constant preserves the relative sensitivity between
        # detectors (which carries sky/SNR information); a per-detector fit would
        # destroy it.  It just happens to land the typical whitened noise near unity
        # for all detectors.  Only the FD_COARSE (BNS) branch uses it; TD/BBH is
        # untouched.
        self._fd_unit_norm = 1.0 / (2.0 * float(delta_f) ** 1.5)

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

    @torch.no_grad()
    def forward(self, input):
        """
        Whiten frequency-domain strain.

        Accepts either a raw tensor (legacy path) or a :class:`GWBatch`
        (state-tracked path).  The behaviour depends on the grid type:

        * **FD_UNIFORM** — whiten → IFFT → strip corrupted edges → return
          ``GWBatch`` with ``TD_UNIFORM`` state (real float32, shape
          ``(B, D, T_valid)``).
        * **FD_COARSE** — whiten at the coarse frequency indices using
          ``batch.coarse_indices`` → return ``GWBatch`` with ``FD_COARSE``
          whitened state (complex, shape ``(B, D, N_coarse)``).
          No IFFT is applied — the non-uniform grid cannot be IFFTed.
        * **Raw tensor** (no GWBatch) — treated as FD_UNIFORM and the raw
          whitened TD tensor is returned for backward compatibility.

        Parameters
        ----------
        input : torch.Tensor or GWBatch
            FD strain ``(B, D, F)`` complex, or a GWBatch wrapping it.

        Returns
        -------
        GWBatch or torch.Tensor
            GWBatch when input is a GWBatch; raw float32 tensor otherwise.
        """
        if isinstance(input, GWBatch):
            return self._forward_batch(input)
        # Legacy raw-tensor path: FD → whitened TD (backward compatible)
        return self._whiten_to_td(input)

    def _whiten_to_td(self, X_fd: torch.Tensor) -> torch.Tensor:
        """Whiten FD strain and convert to valid TD float32."""
        X_white = X_fd * self.whitening.unsqueeze(0)
        x_td    = torch.fft.irfft(X_white, dim=-1, norm="forward") * self.delta_f
        return self.remove_corrupted(x_td)

    def _forward_batch(self, batch: GWBatch) -> GWBatch:
        if batch.state.grid == Grid.FD_COARSE:
            # Non-uniform grid: whiten at the exact coarse indices only.
            # coarse_indices are integer offsets into the full 0→Nyquist
            # whitening buffer — guaranteed to be exact integer multiples of
            # delta_f, so no interpolation is needed.
            idx = batch.coarse_indices                         # (N_coarse,)
            whitening_coarse = self.whitening[:, idx]          # (D, N_coarse)
            # Stay in FD: apply the unit-variance factor the (skipped) IFFT would
            # have provided, so the coarse data is O(1) like the BBH TD path.
            X_white = batch.data * whitening_coarse.unsqueeze(0) * self._fd_unit_norm
            new_state = batch.state.after_whiten()
            return GWBatch(X_white, new_state, batch.freqs, batch.coarse_indices)

        # FD_UNIFORM: existing whiten → IFFT → strip path, wrapped in GWBatch
        x_td      = self._whiten_to_td(batch.data)
        new_state = batch.state.after_whiten().after_ifft()
        return GWBatch(x_td, new_state, freqs=None, coarse_indices=None)
