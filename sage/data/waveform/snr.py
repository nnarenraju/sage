#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

    snr = OptimalSNREstimator(
        asds=fiducial_asds,     # (D, F)
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
from sage.data.asd import get_fiducial_asds


class OptimalSNREstimator(torch.nn.Module):
    """
    Fast batched optimal matched-filter SNR estimator (equivalent to PyCBC ``sigmasq``).

    Computes the optimal (whitened) SNR for a batch of frequency-domain
    detector-projected waveforms using fiducial ASDs loaded from disk.  The
    integration is performed as:

    .. math::

        \\rho^2 = 4 \\Delta f \\sum_f \\frac{|h(f)|^2}{S_n(f)}

    for each detector, summed over detectors for the network SNR.

    Attributes
    ----------
    asds : torch.Tensor, shape ``(1, D, F)``
        Amplitude spectral densities per detector, as stored on disk --
        the strain is divided by them once, not by their square root.
    mask : torch.Tensor or None, shape ``(1, 1, F)``
        Pre-computed frequency mask for ``[f_low, f_high]`` integration band.
    delta_f : float
        Frequency bin spacing in Hz.

    Expected input shapes
    ---------------------
    h   : ``(B, D, F)`` complex tensor — detector-projected FD waveforms.
    """

    def __init__(self):
        """
        Parameters
        ----------
        asd : (D, F) tensor
            Fiducial ASD per detector
        delta_f : float
        f_low, f_high : float or None
            Frequency cutoffs
        """

        super().__init__()

        # Shared config
        cfg = get_cfg()
        data_cfg = get_data_cfg()

        self.device = cfg.device
        # The projected FD waveform lives on the PADDED grid (padded_length_in_s),
        # so its bin spacing is padded_delta_f = 1/(sample_length + 2*padding).
        # Using the unpadded data_cfg.delta_f (1/sample_length) would mis-scale the
        # optimal-SNR integral AND shift the frequency mask onto the wrong bins.
        self.delta_f = data_cfg.padded_delta_f

        # store the ASD once (broadcast ready)
        self.asds = get_fiducial_asds()
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
        asd : real tensor (D, F)
            One-sided ASD for each detector
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
    def forward(self, signal_batch: Tensor):
        """
        Rescale signals to target SNR.

        Args:
            signal_batch: shape [B, L] or [B, C, L]
        Returns:
            rescaled_signal_batch: same shape as input, shape (B, ...)
            scale: (B,) float tensor — per-sample amplitude scale factors
                   (hf_new = hf_old * scale, so distance_new = distance_old / scale)
        """
        B = signal_batch.size(0)
        device = signal_batch.device

        # Compute current network-optimal SNR
        rho_net, _ = self.snr_estimator(signal_batch)  # [B]

        # Sample target SNRs (already float tensor)
        target_rho = self.target_snr_sampler(B).to(device)

        # Compute scaling factors safely
        scale = target_rho.div(rho_net + 1e-12)  # [B]

        return signal_batch * scale[:, None, None], scale


class MatchedFilterSNRRescaler(torch.nn.Module):
    r"""
    Rescale injected signals to a target *matched-filter* network SNR measured
    against the actual noise realisation they are injected into.

    Where :class:`OptimalSNRRescaler` rescales a waveform to a target *optimal*
    SNR from the fiducial ASD alone (pre-noise, expected over noise), this
    rescaler runs AFTER the signal has been paired with its noise window, so it
    uses the real noise. For a signal injected with known parameters (the exact
    template ``h`` at known coalescence time and phase), the coherent
    matched-filter network SNR of ``d = a*h + n`` is

    .. math::

        \rho_\mathrm{mf} = a\,\rho_\mathrm{opt}(h)
                           + \mathrm{Re}\langle n | \hat h \rangle ,
        \qquad \hat h = h / \rho_\mathrm{opt}(h).

    The noise projection :math:`\mathrm{Re}\langle n|\hat h\rangle` does not
    depend on the amplitude ``a`` (``\hat h`` is unit-norm), so solving for the
    amplitude that hits a sampled target :math:`\rho_\star` is a single
    closed-form scale -- no re-injection and no iteration:

    .. math::

        a = \bigl(\rho_\star - \mathrm{Re}\langle n|\hat h\rangle\bigr)
            / \rho_\mathrm{opt}(h).

    :math:`\rho_\mathrm{opt}` and the noise projection are computed once from the
    reference waveform and its paired noise; ``d = a*h + n`` is then formed a
    single time downstream.

    Rationale
    ---------
    Optimal SNR is the *expected* SNR over noise realisations; the realised
    matched-filter SNR fluctuates by :math:`\mathrm{Re}\langle n|\hat h\rangle`
    (unit-variance Gaussian for stationary noise whitened by its own PSD,
    heavier-tailed for real, glitchy or mis-whitened noise). Rescaling to
    optimal SNR therefore leaks a fraction of injected signals whose realised
    SNR is below threshold (undetectable). Rescaling to matched-filter SNR
    removes that leakage: every injected signal sits at (or, in adverse noise,
    above) the target.

    All inner products reuse :class:`OptimalSNREstimator` verbatim -- the same
    fiducial ASD, padded ``delta_f``, DFT-convention ``/delta_f``, ``4 delta_f``
    integration measure and ``[f_low, f_high)`` band -- so the auto-term
    (``rho_opt``) and cross-term (``<n|h>``) are on identical footing. The
    projection is the phase-*coherent* (known-phase) matched filter, which is
    faithful to a known injection and is a lower bound on the phase-marginalised
    ``|z|`` a blind search would report -- so rescaling it to >= target keeps the
    search statistic at least as high.

    Parameters
    ----------
    target_snr_sampler : callable(batch_size) -> Tensor, shape (batch_size,)
        Samples target *matched-filter* network SNRs (e.g. :class:`HalfNorm`).
    """

    def __init__(self, target_snr_sampler: Callable[[int], Tensor]):
        super().__init__()
        self.snr_estimator = OptimalSNREstimator()
        self.target_snr_sampler = target_snr_sampler

    def _noise_projection(self, noise: Tensor, signal: Tensor) -> Tensor:
        """Network noise--template inner product ``Re<n|h>`` (before ``/rho_opt``).

        Uses the estimator's fiducial whitening so the cross-term matches the
        auto-term ``rho_opt^2 = <h|h>`` bin-for-bin. ``noise`` and ``signal`` are
        FD ``(S, D, F)`` on the padded grid; returns a real ``(S,)`` tensor,
        integrated over frequency then summed coherently over detectors.
        """
        est = self.snr_estimator
        # Whiten both exactly as rho_opt does: (x/df)/asd recovers the physical
        # continuous FT divided by the ASD (see OptimalSNREstimator.forward).
        n_white = (noise / est.delta_f) / est.asds
        h_white = (signal / est.delta_f) / est.asds
        # Re(n_white * conj(h_white)) -- the known-phase coherent projection.
        cross = n_white.real * h_white.real + n_white.imag * h_white.imag
        if est.mask is not None:
            cross = cross * est.mask.to(cross.dtype)
        # 4 df integral over frequency, then coherent sum over detectors -> (S,)
        cross_det = 4.0 * est.delta_f * cross.sum(dim=-1)   # (S, D)
        return cross_det.sum(dim=1)                          # (S,)

    @torch.no_grad()
    def forward(self, signal_batch: Tensor, noise_batch: Tensor):
        """
        Rescale each injected signal to a sampled target matched-filter SNR in
        its own noise window.

        Parameters
        ----------
        signal_batch : complex tensor (S, D, F)
            Reference detector-projected FD waveforms (physical amplitude).
        noise_batch : complex tensor (S, D, F)
            The FD noise realisations the signals are injected into, row-aligned
            with ``signal_batch`` (``noise_data[idx]``). Already recoloured when
            recolour is on -- the actual noise the network sees.

        Returns
        -------
        rescaled_signal_batch : complex tensor (S, D, F)
            ``signal_batch * a`` on the same grid and dtype.
        a : real tensor (S,)
            Per-signal amplitude scale (``hf_new = hf_old * a``, so
            ``distance_new = distance_old / a``).
        """
        S = signal_batch.size(0)
        device = signal_batch.device

        rho_opt, _ = self.snr_estimator(signal_batch)                  # (S,)
        proj = self._noise_projection(noise_batch, signal_batch)        # Re<n|h>
        # Re<n|h_hat> = Re<n|h> / rho_opt (amplitude-independent noise term).
        proj = proj / (rho_opt + 1e-12)                                 # (S,)

        target = self.target_snr_sampler(S).to(device=device, dtype=rho_opt.dtype)

        # a = (target - Re<n|h_hat>) / rho_opt.
        # Adverse-noise guard: if the noise already projects at/above the target
        # along the template (numerator <= 0 -- possible for real glitchy or
        # mis-whitened noise, a >5 sigma event for stationary Gaussian), fall
        # back to optimal SNR = target so the injection stays a positive,
        # detectable signal instead of vanishing or flipping sign.
        num = target - proj
        a = torch.where(num > 0, num, target) / (rho_opt + 1e-12)       # (S,)
        a = a.to(signal_batch.real.dtype)

        return signal_batch * a[:, None, None], a


class ChiSquaredReweightedSNRRescaler(torch.nn.Module):
    r"""
    Rescale injected signals to a target **PyCBC re-weighted "new SNR"** measured
    in the real noise -- the χ²-discriminated statistic a real matched-filter
    search actually ranks on. Faithful to PyCBC (``pycbc.vetoes.chisq.power_chisq``
    + ``pycbc.events.ranking.newsnr`` + ``QuadratureSumStatistic``).

    Raw matched-filter SNR is inflated by non-Gaussian transients: a glitch
    aligned with the template lifts ρ, so a weak signal on a glitch looks loud
    while a real search would down-weight it via the Allen χ² test and miss it.
    Targeting the re-weighted SNR removes that leakage -- in glitchy windows the
    signal must be injected louder to survive the χ² penalty, so the "signal"
    class stays genuinely detectable.

    Method (all per detector, evaluated at the known injection time; inner
    products reuse :class:`OptimalSNREstimator` verbatim)
    ------------------------------------------------------------------------
    * Single-detector SNR ``|z_d| = |a·ρ_opt_d + ν_d|`` (PyCBC uses |z|), with
      ``ν_d = <n_d|ĥ_d>`` the complex noise projection (amplitude-independent).
    * Allen power-χ²: split the template into ``num_bins`` equal-power frequency
      bins (PyCBC ``power_chisq_bins``: cumulative in-band power, split at
      ``j·σ²/p``), then ``χ²_d = p·Σ_i|z_{d,i}|² − |z_d|²`` (PyCBC's exact
      ``chisq·num_bins − |snr|²`` form), dof ``= 2p−2``, reduced ``χ²ᵣ = χ²/dof``.
      Because the injected template is the χ² template, the signal distributes
      equally across bins and CANCELS in ``z_{d,i} − z_d/p`` -- so **χ² depends
      only on the noise**, computed here from ``ν`` alone and identical to PyCBC
      ``power_chisq`` on ``a·h + n``.
    * Re-weight (PyCBC ``newsnr``, q=6, n=2): ``ρ̂_d = |z_d| / g_d`` with
      ``g_d = [½(1 + χ²ᵣ³)]^{1/6}`` if ``χ²ᵣ > 1`` else ``1``. ``g_d`` is
      amplitude-independent (χ² is).
    * Network (PyCBC ``QuadratureSumStatistic``): ``ρ̂_net² = Σ_d ρ̂_d²``.

    Solve
    -----
    ``ρ̂_net²(a) = Σ_d |a·ρ_opt_d + ν_d|² / g_d² = A a² + B a + C``, set ``=
    target²`` and take the positive root -- one closed-form quadratic, no
    iteration (``g_d`` and ``ν_d`` are amplitude-independent). Adverse-noise
    fallback: if there is no positive root (the re-weighted noise alone already
    reaches the target), rescale to optimal network SNR = target so the injection
    stays positive and detectable.

    Parameters
    ----------
    target_snr_sampler : callable(batch_size) -> Tensor, shape (batch_size,)
        Samples target *network re-weighted* SNRs (e.g. :class:`HalfNorm`).
    num_bins : int
        Number of equal-power χ² bins ``p`` (PyCBC default region ~16).
    """

    def __init__(self, target_snr_sampler: Callable[[int], Tensor], num_bins: int = 16):
        super().__init__()
        self.snr_estimator = OptimalSNREstimator()
        self.target_snr_sampler = target_snr_sampler
        self.num_bins = int(num_bins)

    def _perdet_stats(self, signal: Tensor, noise: Tensor):
        """Per-detector (ρ_opt, ν=<n|ĥ> complex, g newsnr-weight, χ²), all at the
        injection time. χ² is amplitude-independent (exact-template cancellation),
        so it is computed from the noise projection alone. Shapes: (S,D) real,
        (S,D) complex, (S,D) real, (S,D) real."""
        est = self.snr_estimator
        p = self.num_bins
        df = est.delta_f
        h_w = (signal / df) / est.asds                      # (S,D,F) complex
        n_w = (noise / df) / est.asds
        mask = est.mask                                     # (1,1,F) or None

        # rho_opt per detector (the verified estimator, same whitening/band)
        _, rho_det = est(signal)                            # (S,D)

        # equal-power chisq bins: PyCBC cumulative in-band power split at j*sig/p.
        power = h_w.real * h_w.real + h_w.imag * h_w.imag   # |h_w|^2 (S,D,F)
        if mask is not None:
            power = power * mask.to(power.dtype)
        cumpow = torch.cumsum(power, dim=-1)                # (S,D,F) monotone
        total = cumpow[..., -1:].clamp_min(1e-30)           # (S,D,1) = in-band sigma^2
        # per-frequency bin label (floor == PyCBC searchsorted edges up to ties);
        # out-of-band bins get zero cross term below so their label is irrelevant.
        binf = torch.floor(cumpow / total * p).clamp_(0, p - 1).to(torch.long)

        # per-bin complex cross term c_i = 4 df sum_{f in bin i} n_w conj(h_w)
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

        # chisq = p*sum_i|nu_i|^2 - |nu|^2  (PyCBC power_chisq form), dof = 2p-2
        chisq = (p * (nu_i.real * nu_i.real + nu_i.imag * nu_i.imag).sum(dim=2)
                 - (nu.real * nu.real + nu.imag * nu.imag)).clamp_min(0.0)
        rchisq = chisq / (2.0 * p - 2.0)
        # PyCBC newsnr down-weight (q=6, n=2): g = [0.5(1+rchisq^3)]^(1/6) if >1
        g = torch.where(
            rchisq > 1.0,
            (0.5 * (1.0 + rchisq * rchisq * rchisq)).pow(1.0 / 6.0),
            torch.ones_like(rchisq),
        )
        return rho_det, nu, g, chisq

    @torch.no_grad()
    def forward(self, signal_batch: Tensor, noise_batch: Tensor):
        """
        Rescale each injected signal to a sampled target network re-weighted SNR
        in its own noise window.

        Parameters
        ----------
        signal_batch : complex tensor (S, D, F)
            Reference detector-projected FD waveforms (physical amplitude).
        noise_batch : complex tensor (S, D, F)
            The FD noise realisations the signals are injected into, row-aligned
            with ``signal_batch`` (``noise_data[idx]``), already recoloured when
            recolour is on.

        Returns
        -------
        rescaled_signal_batch : complex tensor (S, D, F) -- ``signal_batch * a``.
        a : real tensor (S,) -- per-signal amplitude scale (distance /= a).
        """
        device = signal_batch.device
        S = signal_batch.size(0)

        rho, nu, g, _ = self._perdet_stats(signal_batch, noise_batch)   # (S,D)...
        g2 = g * g

        # network re-weighted SNR^2(a) = sum_d |a rho_d + nu_d|^2 / g_d^2
        #                              = A a^2 + B a + C  (quadratic in a)
        A = (rho * rho / g2).sum(dim=1)                     # (S,)
        B = (2.0 * rho * nu.real / g2).sum(dim=1)           # (S,)
        C = ((nu.real * nu.real + nu.imag * nu.imag) / g2).sum(dim=1)   # (S,)

        target = self.target_snr_sampler(S).to(device=device, dtype=A.dtype)
        disc = B * B + 4.0 * A * (target * target - C)
        a_quad = (-B + torch.sqrt(disc.clamp_min(0.0))) / (2.0 * A + 1e-30)

        # Adverse-noise fallback: no positive root (disc<0 or a<=0 -- the
        # re-weighted noise alone already reaches target) -> rescale to optimal
        # network SNR = target so the injection stays positive and detectable.
        rho_net = torch.sqrt((rho * rho).sum(dim=1))
        a_fallback = target / (rho_net + 1e-12)
        bad = (disc < 0) | (a_quad <= 0) | torch.isnan(a_quad)
        a = torch.where(bad, a_fallback, a_quad).to(signal_batch.real.dtype)

        return signal_batch * a[:, None, None], a
