#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimal SNR of detector-frame waveforms against fiducial ASDs, and
rescalers of signals to target optimal, matched-filter or re-weighted
SNRs.

Filename        : snr.py
Description     : Short description of the file

Created on 2026-02-16 11:14:49

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation:

    snr = OptimalSNREstimator(fiducial_asds, window, f_low=20.0, device="cuda")

    rho_net, rho_det = snr(h_batch)

"""

import torch

from torch import Tensor
from typing import Callable

from sage.core.geometry import Window


class OptimalSNREstimator(torch.nn.Module):
    """
    Batched optimal SNR against fiducial ASDs, as PyCBC ``sigmasq``.

    ``rho^2 = 4 delta_f sum_f |h / delta_f|^2 / ASD^2`` per detector
    over the band, summed over detectors for the network SNR.

    Parameters
    ----------
    asds : torch.Tensor, shape ``(D, F)``
        Fiducial ASD per detector.
    window : Window
        Window geometry; the waveforms live on its padded grid.
    f_low : float
        Lower edge of the band in Hz; the upper edge is the Nyquist
        frequency.
    device : str or torch.device
        Device of the mask.

    Attributes
    ----------
    asds : torch.Tensor, shape ``(1, D, F)``
        The fiducial ASDs; the strain is divided by them, not by their
        square root.
    mask : torch.Tensor, shape ``(1, 1, F)``
        Mask of the band ``[f_low, f_high)``.
    delta_f : float
        Frequency spacing in Hz.
    """

    def __init__(self, asds: Tensor, window: Window, f_low: float, device):

        super().__init__()

        self.device = device
        self.delta_f = window.delta_f_in_hz
        self.asds = asds.unsqueeze(0)  # (1, D, F)

        F = self.asds.shape[-1]
        self.mask = self._make_frequency_mask(F, f_low, window.sample_rate_in_hz // 2)

    def _make_frequency_mask(self, F, f_low, f_high):
        """
        Mask of the bins ``ceil(f_low / df)`` to ``floor(f_high / df)``.

        The upper bin is excluded.

        Parameters
        ----------
        F : int
            Number of frequency bins.
        f_low : float
            Lower edge in Hz.
        f_high : float
            Upper edge in Hz.

        Returns
        -------
        torch.Tensor, shape ``(1, 1, F)``, float64
            1 in the band, 0 outside.
        """
        k_low = int(torch.ceil(torch.tensor(f_low / self.delta_f)).item())
        k_high = int(torch.floor(torch.tensor(f_high / self.delta_f)).item())

        mask = torch.zeros(F, dtype=torch.float64, device=self.device)
        mask[k_low:k_high] = 1.0
        return mask.view(1, 1, F)  # broadcastable

    def forward(self, h, asds=None):
        """
        Optimal SNR of each waveform, per detector and over the network.

        Parameters
        ----------
        h : torch.Tensor, shape ``(B, D, F)``, complex
            Detector-frame frequency-domain strain.
        asds : torch.Tensor, shape ``(B, D, F)``, or None
            ASD to integrate against, per row; None uses the fiducial
            ASDs.

        Returns
        -------
        rho_net : torch.Tensor, shape ``(B,)``
            Network optimal SNR.
        rho_det : torch.Tensor, shape ``(B, D)``
            Optimal SNR per detector.
        """

        # Divide by the ASD instead of squaring it, (B,D,F)
        h_white = (h / self.delta_f) / (self.asds if asds is None else asds)

        # |h|^2
        power = h_white.real * h_white.real + h_white.imag * h_white.imag

        if self.mask is not None:
            power *= self.mask

        # integrate frequency (B,D)
        rho2_det = 4.0 * self.delta_f * power.sum(dim=-1)
        # network combine (B,1)
        rho2_net = rho2_det.sum(dim=1, keepdim=True)

        # Shape (B,D)
        rho_det = torch.sqrt(rho2_det)
        # Shape (B,)
        rho_net = torch.sqrt(rho2_net).squeeze(-1)

        return rho_net, rho_det


class OptimalSNRRescaler(torch.nn.Module):
    """
    Rescale signals to target network optimal SNRs.

    Parameters
    ----------
    target_snr_sampler : callable
        ``batch_size -> Tensor`` of target network SNRs.
    asds, window, f_low, device
        Passed to :class:`OptimalSNREstimator`.
    """

    def __init__(
        self,
        target_snr_sampler: Callable[[int], Tensor],
        asds: Tensor,
        window: Window,
        f_low: float,
        device,
    ):
        super().__init__()
        self.snr_estimator = OptimalSNREstimator(asds, window, f_low, device)
        self.target_snr_sampler = target_snr_sampler

    @torch.no_grad()
    def forward(self, signal_batch: Tensor):
        """
        Rescale each signal to a sampled target SNR.

        Parameters
        ----------
        signal_batch : torch.Tensor, shape ``(B, D, F)``, complex
            Detector-frame frequency-domain strain.

        Returns
        -------
        rescaled_signal_batch : torch.Tensor, shape ``(B, D, F)``
            ``signal_batch * scale``.
        scale : torch.Tensor, shape ``(B,)``
            Amplitude scale of each signal; the distance scales by
            ``1 / scale``.
        """
        B = signal_batch.size(0)
        device = signal_batch.device

        rho_net, _ = self.snr_estimator(signal_batch)  # [B]

        target_rho = self.target_snr_sampler(B).to(device)

        # 1e-12 guards a zero-SNR signal
        scale = target_rho.div(rho_net + 1e-12)  # [B]

        return signal_batch * scale[:, None, None], scale


class MatchedFilterSNRRescaler(torch.nn.Module):
    """
    Rescale injected signals to a target matched-filter network SNR.

    Measured in the noise each signal is injected into, with the
    known-phase inner products of :class:`OptimalSNREstimator`:
    ``a = (target - Re<n|h_hat>) / rho_opt``, ``h_hat = h / rho_opt``.

    Parameters
    ----------
    target_snr_sampler : callable
        ``batch_size -> Tensor`` of target matched-filter network SNRs.
    asds, window, f_low, device
        Passed to :class:`OptimalSNREstimator`.
    """

    def __init__(
        self,
        target_snr_sampler: Callable[[int], Tensor],
        asds: Tensor,
        window: Window,
        f_low: float,
        device,
    ):
        super().__init__()
        self.snr_estimator = OptimalSNREstimator(asds, window, f_low, device)
        self.target_snr_sampler = target_snr_sampler

    def _noise_projection(self, noise: Tensor, signal: Tensor) -> Tensor:
        """
        Network noise-template inner product ``Re<n|h>``.

        Whitened, banded and integrated as :class:`OptimalSNREstimator`,
        then summed over detectors.

        Parameters
        ----------
        noise : torch.Tensor, shape ``(S, D, F)``, complex
            Noise on the padded grid.
        signal : torch.Tensor, shape ``(S, D, F)``, complex
            Signals on the padded grid.

        Returns
        -------
        torch.Tensor, shape ``(S,)``
            ``Re<n|h>``, before the division by ``rho_opt``.
        """
        est = self.snr_estimator
        # Whitened as in OptimalSNREstimator.forward
        n_white = (noise / est.delta_f) / est.asds
        h_white = (signal / est.delta_f) / est.asds
        # Re(n_white * conj(h_white)), the known-phase projection
        cross = n_white.real * h_white.real + n_white.imag * h_white.imag
        if est.mask is not None:
            cross = cross * est.mask.to(cross.dtype)
        # 4 df integral over frequency, then sum over detectors
        cross_det = 4.0 * est.delta_f * cross.sum(dim=-1)   # (S, D)
        return cross_det.sum(dim=1)                          # (S,)

    @torch.no_grad()
    def forward(self, signal_batch: Tensor, noise_batch: Tensor):
        """
        Rescale each signal to a sampled target matched-filter SNR.

        Where the noise alone reaches the target, the optimal SNR is set
        to the target instead.

        Parameters
        ----------
        signal_batch : torch.Tensor, shape ``(S, D, F)``, complex
            Detector-frame waveforms at physical amplitude.
        noise_batch : torch.Tensor, shape ``(S, D, F)``, complex
            Noise each signal is injected into, row-aligned with
            ``signal_batch``.

        Returns
        -------
        rescaled_signal_batch : torch.Tensor, shape ``(S, D, F)``
            ``signal_batch * a``.
        a : torch.Tensor, shape ``(S,)``
            Amplitude scale of each signal; the distance scales by
            ``1 / a``.
        """
        S = signal_batch.size(0)
        device = signal_batch.device

        rho_opt, _ = self.snr_estimator(signal_batch)                  # (S,)
        proj = self._noise_projection(noise_batch, signal_batch)        # Re<n|h>
        # Re<n|h_hat> = Re<n|h> / rho_opt, independent of the amplitude
        proj = proj / (rho_opt + 1e-12)                                 # (S,)

        target = self.target_snr_sampler(S).to(device=device, dtype=rho_opt.dtype)

        # a = (target - Re<n|h_hat>) / rho_opt; where the numerator is
        # <= 0, the optimal SNR is set to the target instead
        num = target - proj
        a = torch.where(num > 0, num, target) / (rho_opt + 1e-12)       # (S,)
        a = a.to(signal_batch.real.dtype)

        return signal_batch * a[:, None, None], a


class ChiSquaredReweightedSNRRescaler(torch.nn.Module):
    """
    Rescale injected signals to a target network re-weighted SNR.

    Per detector, PyCBC's newsnr (q=6, n=2) from the power chi^2 of
    ``num_bins`` equal-power bins, summed in quadrature over detectors;
    the scale is the positive root of a quadratic in the amplitude.

    Parameters
    ----------
    target_snr_sampler : callable
        ``batch_size -> Tensor`` of target network re-weighted SNRs.
    asds, window, f_low, device
        Passed to :class:`OptimalSNREstimator`.
    num_bins : int
        Number of equal-power chi^2 bins ``p``.
    """

    def __init__(
        self,
        target_snr_sampler: Callable[[int], Tensor],
        asds: Tensor,
        window: Window,
        f_low: float,
        device,
        num_bins: int = 16,
    ):
        super().__init__()
        self.snr_estimator = OptimalSNREstimator(asds, window, f_low, device)
        self.target_snr_sampler = target_snr_sampler
        self.num_bins = int(num_bins)

    def _perdet_stats(self, signal: Tensor, noise: Tensor):
        """
        Per-detector statistics at the injection time.

        chi^2 depends only on the noise projection: the injected
        template is the chi^2 template, so the signal cancels.

        Parameters
        ----------
        signal : torch.Tensor, shape ``(S, D, F)``, complex
            Signals on the padded grid.
        noise : torch.Tensor, shape ``(S, D, F)``, complex
            Noise on the padded grid.

        Returns
        -------
        rho_det : torch.Tensor, shape ``(S, D)``
            Optimal SNR.
        nu : torch.Tensor, shape ``(S, D)``, complex
            Noise projection ``<n|h_hat>``.
        g : torch.Tensor, shape ``(S, D)``
            newsnr down-weight, ``newsnr = |z| / g``.
        chisq : torch.Tensor, shape ``(S, D)``
            Power chi^2 with ``2 p - 2`` degrees of freedom.
        """
        est = self.snr_estimator
        p = self.num_bins
        df = est.delta_f
        h_w = (signal / df) / est.asds                      # (S,D,F) complex
        n_w = (noise / df) / est.asds
        mask = est.mask                                     # (1,1,F) or None

        # rho_opt per detector
        _, rho_det = est(signal)                            # (S,D)

        # Equal-power chisq bins: PyCBC's cumulative in-band power split
        # at j*sig/p
        power = h_w.real * h_w.real + h_w.imag * h_w.imag   # |h_w|^2 (S,D,F)
        if mask is not None:
            power = power * mask.to(power.dtype)
        cumpow = torch.cumsum(power, dim=-1)                # (S,D,F) monotone
        total = cumpow[..., -1:].clamp_min(1e-30)           # (S,D,1) = in-band sigma^2
        # Bin of each frequency (floor, PyCBC's searchsorted edges up to
        # ties); out-of-band frequencies carry no cross term
        binf = torch.floor(cumpow / total * p).clamp_(0, p - 1).to(torch.long)

        # Per-bin cross term c_i = 4 df sum_{f in bin i} n_w conj(h_w)
        cross = n_w * torch.conj(h_w)                       # (S,D,F) complex
        if mask is not None:
            cross = cross * mask.to(cross.real.dtype)
        cross = (4.0 * df) * cross
        S, D, F = cross.shape
        ci_r = torch.zeros(S, D, p, dtype=cross.real.dtype, device=cross.device)
        ci_i = torch.zeros(S, D, p, dtype=cross.real.dtype, device=cross.device)
        ci_r.scatter_add_(2, binf, cross.real)
        ci_i.scatter_add_(2, binf, cross.imag)
        c_i = torch.complex(ci_r, ci_i)                     # (S,D,p) = <n|h>_bin
        c_tot = c_i.sum(dim=2)                              # (S,D)   = <n|h>

        rho = rho_det + 1e-12
        nu_i = c_i / rho.unsqueeze(-1)                      # (S,D,p) = <n|h_hat>_bin
        nu = c_tot / rho                                    # (S,D)   = <n|h_hat>

        # chisq = p*sum_i|nu_i|^2 - |nu|^2 (PyCBC power_chisq), dof 2p-2
        chisq = (p * (nu_i.real * nu_i.real + nu_i.imag * nu_i.imag).sum(dim=2)
                 - (nu.real * nu.real + nu.imag * nu.imag)).clamp_min(0.0)
        rchisq = chisq / (2.0 * p - 2.0)
        # PyCBC newsnr (q=6, n=2): g = [0.5(1+rchisq^3)]^(1/6) if > 1
        g = torch.where(
            rchisq > 1.0,
            (0.5 * (1.0 + rchisq * rchisq * rchisq)).pow(1.0 / 6.0),
            torch.ones_like(rchisq),
        )
        return rho_det, nu, g, chisq

    @torch.no_grad()
    def forward(self, signal_batch: Tensor, noise_batch: Tensor):
        """
        Rescale each signal to a sampled target network re-weighted SNR.

        Without a positive root, the optimal network SNR is set to the
        target instead.

        Parameters
        ----------
        signal_batch : torch.Tensor, shape ``(S, D, F)``, complex
            Detector-frame waveforms at physical amplitude.
        noise_batch : torch.Tensor, shape ``(S, D, F)``, complex
            Noise each signal is injected into, row-aligned with
            ``signal_batch``.

        Returns
        -------
        rescaled_signal_batch : torch.Tensor, shape ``(S, D, F)``
            ``signal_batch * a``.
        a : torch.Tensor, shape ``(S,)``
            Amplitude scale of each signal; the distance scales by
            ``1 / a``.
        """
        device = signal_batch.device
        S = signal_batch.size(0)

        rho, nu, g, _ = self._perdet_stats(signal_batch, noise_batch)   # (S,D)...
        g2 = g * g

        # Network re-weighted SNR^2(a)
        #   = sum_d |a rho_d + nu_d|^2 / g_d^2 = A a^2 + B a + C
        A = (rho * rho / g2).sum(dim=1)                     # (S,)
        B = (2.0 * rho * nu.real / g2).sum(dim=1)           # (S,)
        C = ((nu.real * nu.real + nu.imag * nu.imag) / g2).sum(dim=1)   # (S,)

        target = self.target_snr_sampler(S).to(device=device, dtype=A.dtype)
        disc = B * B + 4.0 * A * (target * target - C)
        a_quad = (-B + torch.sqrt(disc.clamp_min(0.0))) / (2.0 * A + 1e-30)

        # Without a positive root (disc < 0 or a <= 0), the optimal
        # network SNR is set to the target instead
        rho_net = torch.sqrt((rho * rho).sum(dim=1))
        a_fallback = target / (rho_net + 1e-12)
        bad = (disc < 0) | (a_quad <= 0) | torch.isnan(a_quad)
        a = torch.where(bad, a_fallback, a_quad).to(signal_batch.real.dtype)

        return signal_batch * a[:, None, None], a
