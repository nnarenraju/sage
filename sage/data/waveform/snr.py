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


class OptimalSNREstimator:
    """
    Fast batched optimal SNR calculator (PyCBC sigmasq equivalent).

    Expected shapes
    ---------------
    h   : (B, D, F) complex64/complex128
    psd : (D, F)    real
    """

    def __init__(self, psds, delta_f, f_low=None, f_high=None, device="cuda"):
        """
        Parameters
        ----------
        psd : (D, F) tensor
            Fiducial PSD per detector
        delta_f : float
        f_low, f_high : float or None
            Frequency cutoffs
        """

        self.device = device
        self.delta_f = float(delta_f)

        # store PSD once (broadcast ready)
        self.psds = psds.to(device)
        self.psds = self.psds.unsqueeze(0)  # (1, D, F)

        # precompute mask once (compile safe)
        self.mask = None
        if f_low is not None and f_high is not None:
            F = psds.shape[-1]
            self.mask = self._make_frequency_mask(F, f_low, f_high)

    def _make_frequency_mask(self, F, f_low, f_high):
        k_low = int(torch.ceil(torch.tensor(f_low / self.delta_f)).item())
        k_high = int(torch.floor(torch.tensor(f_high / self.delta_f)).item())

        mask = torch.zeros(F, dtype=self.psds.dtype, device=self.device)
        mask[k_low:k_high] = 1.0
        print(mask.sum())
        return mask.view(1, 1, F)  # broadcastable

    @torch.compile(fullgraph=True)
    def __call__(self, h):
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

        # |h|^2
        power = h.real * h.real + h.imag * h.imag  # (B,D,F)

        # PSD weighting
        weighted = power / self.psds

        # apply mask if exists
        if self.mask is not None:
            weighted *= self.mask

        # integrate frequency
        rho2_det = 4.0 * self.delta_f * weighted.sum(dim=-1)  # (B,D)

        # network combine
        rho2_net = rho2_det.sum(dim=1, keepdim=True)  # (B,1)

        rho_det = torch.sqrt(rho2_det).unsqueeze(-1)  # (B,D,1)
        rho_net = torch.sqrt(rho2_net)  # (B,1)

        return rho_net, rho_det
