#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : snr.py
Description     : Short description of the file

Created on 2026-02-16 11:14:49

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    snr = OptimalSNREstimator(
        psds=fiducial_psds,     # (D, F)
        delta_f=delta_f,
        f_low=20.0,
        f_high=1024.0,
        device="cuda"
    )

    rho_net, rho_det = snr(h_batch)

"""

# Packages
import torch

from torch import Tensor
from typing import Callable

# LOCAL
from sage.core.config import get_cfg, get_data_cfg
from sage.data.psd import get_fiducial_psds


class OptimalSNREstimator(torch.nn.Module):
    """
    Fast batched optimal SNR calculator (PyCBC sigmasq equivalent).

    Expected shapes
    ---------------
    h   : (B, D, F) complex64/complex128
    psd : (D, F)    real
    """

    def __init__(self):
        """
        Parameters
        ----------
        psd : (D, F) tensor
            Fiducial PSD per detector
        delta_f : float
        f_low, f_high : float or None
            Frequency cutoffs
        """

        super().__init__()

        # Shared config
        cfg = get_cfg()
        data_cfg = get_data_cfg()

        self.device = cfg.device
        self.delta_f = data_cfg.delta_f

        # store PSD once (broadcast ready)
        self.asds = get_fiducial_psds()
        self.asds = self.asds.unsqueeze(0)  # (1, D, F)

        # precompute mask once (compile safe)
        self.mask = None
        f_low = data_cfg.signal_low_frequency_cutoff
        f_high = data_cfg.sample_rate // 2
        if f_low is not None and f_high is not None:
            F = self.asds.shape[-1]
            self.mask = self._make_frequency_mask(F, f_low, f_high)

    def _make_frequency_mask(self, F, f_low, f_high):
        k_low = int(torch.ceil(torch.tensor(f_low / self.delta_f)).item())
        k_high = int(torch.floor(torch.tensor(f_high / self.delta_f)).item())

        # mask = torch.zeros(F, dtype=self.psds.dtype, device=self.device)
        mask = torch.zeros(F, dtype=torch.float64, device=self.device)
        mask[k_low:k_high] = 1.0
        return mask.view(1, 1, F)  # broadcastable

    def forward(self, h):
        """
        Batched optimal SNR for multi-detector frequency-domain waveforms.

        Parameters
        ----------
        h : complex tensor (B, D, F)
            Detector-projected frequency-domain strain
        psd : real tensor (D, F)
            One-sided PSD for each detector
        delta_f : float
            Frequency spacing
        mask : optional bool tensor (F,)
            Frequency mask for f_low / f_high cutoffs

        Returns
        -------
        rho_net : (B, 1)
            Network optimal SNR
        rho_det : (B, D, 1)
            Per-detector optimal SNR
        """

        # whiten waveform instead of squaring ASD (B,D,F)
        h_white = (h / self.delta_f) / self.asds

        # |h|^2
        power = h_white.real * h_white.real + h_white.imag * h_white.imag

        # apply mask if exists
        if self.mask is not None:
            power *= self.mask

        # integrate frequency (B,D)
        rho2_det = 4.0 * self.delta_f * power.sum(dim=-1)
        # network combine (B,1)
        rho2_net = rho2_det.sum(dim=1, keepdim=True)

        # Shape (B,D,1)
        rho_det = torch.sqrt(rho2_det)
        # Shape (B,1)
        rho_net = torch.sqrt(rho2_net).squeeze(-1)

        return rho_net, rho_det


class OptimalSNRRescaler(torch.nn.Module):
    """
    Rescales a batch of signals to match target SNRs.

    Args:
        snr_estimator: instance of OptimalSNREstimator
        target_snr_sampler: callable(batch_size) -> Tensor of target SNRs
    """

    def __init__(self, target_snr_sampler: Callable[[int], Tensor]):
        super().__init__()
        self.snr_estimator = OptimalSNREstimator()
        self.target_snr_sampler = target_snr_sampler

    @torch.no_grad()
    def forward(self, signal_batch: Tensor) -> Tensor:
        """
        Rescale signals to target SNR.

        Args:
            signal_batch: shape [B, L] or [B, C, L]
        Returns:
            rescaled_signal_batch: same shape as input
        """
        B = signal_batch.size(0)
        device = signal_batch.device

        # Compute current network-optimal SNR
        rho_net, _ = self.snr_estimator(signal_batch)  # [B]

        # Sample target SNRs (already float tensor)
        target_rho = self.target_snr_sampler(B).to(device)

        # Compute scaling factors safely
        scale = target_rho.div(rho_net + 1e-12)  # [B]

        return signal_batch * scale[:, None, None]
