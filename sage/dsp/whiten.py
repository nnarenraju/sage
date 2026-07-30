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
import numpy as np
import torch
import matplotlib.pyplot as plt

# LOCAL
from sage.data.psd import get_fiducial_psds
from sage.core.config import get_cfg, get_data_cfg
from sage.core.pipeline import GWBatch, Grid, ProcessingState


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

        # Exp 1: optional suppress-only FFT per-bin line notch. After whitening, bins
        # flagged as spectral lines (fiducial ASD >> its running-median floor) have
        # their magnitude pulled DOWN to the local floor (gain<=1, phase kept) -- it
        # removes the residual line power the fiducial's median-based envelope leaves
        # (esp. L1 wandering lines) and, because gain<=1, can never amplify a segment.
        # Signals carry ~no power in the ~1.5%-of-band narrow line bins -> zero SNR loss.
        # Default OFF: configs that don't set `use_line_notch` are byte-for-byte unchanged.
        self._line_notch_on = bool(getattr(cfg, "use_line_notch", False))
        if self._line_notch_on:
            self._build_line_notch(fiducial_psds, cfg, data_cfg)

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

    def _build_line_notch(self, fiducial_asds, cfg, data_cfg, win=201, thr=1.8, k_ref=32):
        """Precompute per-detector line-bin indices + local-floor reference indices.

        A bin is a "line" when the fiducial ASD exceeds its running-median floor by
        ``thr``, within ``[signal_low_frequency_cutoff, Nyquist]``. The union of O3a+O3b
        lines is captured automatically because the fiducial here is the combined max(A,B).
        """
        from scipy.signal import medfilt

        asds = fiducial_asds.detach().cpu().numpy().astype(np.float64)   # (D, F)
        D, Fbins = asds.shape
        df = float(self.sample_rate) / self.seq_len
        freqs = np.arange(Fbins) * df
        inband = (freqs >= float(data_cfg.signal_low_frequency_cutoff)) & (freqs <= self.sample_rate / 2)

        self._notch_line_idx, self._notch_ref_idx = [], []
        counts = []
        for d in range(D):
            floor = np.maximum(medfilt(asds[d], win), 1e-40)
            mask = (asds[d] / floor > thr) & inband
            line_bins = np.where(mask)[0]
            nonline = np.where((~mask) & inband)[0]
            ref = np.empty((len(line_bins), k_ref), dtype=np.int64)
            for j, b in enumerate(line_bins):
                pos = int(np.searchsorted(nonline, b))
                lo, hi = max(0, pos - k_ref), min(len(nonline), pos + k_ref)
                cand = nonline[lo:hi]
                sel = cand[np.argsort(np.abs(cand - b))[:k_ref]]
                if len(sel) < k_ref:
                    sel = np.pad(sel, (0, k_ref - len(sel)), mode="edge")
                ref[j] = sel
            self._notch_line_idx.append(torch.from_numpy(line_bins).to(self.device))
            self._notch_ref_idx.append(torch.from_numpy(ref).to(self.device))
            counts.append(len(line_bins))
        dets = getattr(cfg, "detectors", [f"d{d}" for d in range(D)])
        print("[FiducialWhitening] line notch ON: "
              + ", ".join(f"{dets[d]}={counts[d]} bins" for d in range(D)))

    def _apply_line_notch(self, X_white: torch.Tensor) -> torch.Tensor:
        """Suppress-only per-bin line notch on whitened FD data (in place, gain<=1)."""
        eps = 1e-20
        for d, (li, ri) in enumerate(zip(self._notch_line_idx, self._notch_ref_idx)):
            if li.numel() == 0:
                continue
            mag = X_white[:, d].abs()                        # (B, F)
            ref = mag[:, ri].median(dim=-1).values           # (B, n_lines)
            cur = mag[:, li]                                 # (B, n_lines)
            gain = torch.clamp(ref / (cur + eps), max=1.0)    # (B, n_lines) in [0,1]
            X_white[:, d, li] = X_white[:, d, li] * gain
        return X_white

    def _whiten_to_td(self, X_fd: torch.Tensor) -> torch.Tensor:
        """Whiten FD strain and convert to valid TD float32."""
        X_white = X_fd * self.whitening.unsqueeze(0)
        if self._line_notch_on:
            X_white = self._apply_line_notch(X_white)
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
