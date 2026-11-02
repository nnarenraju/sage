#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-astrophysical (decoherent) two-detector sample generation: hard background
pairs that look time-coincident but are physically inconsistent, used to teach a
detection network that coincidence alone is not detection.

A run enables it with ``signal.non_astrophysical``; :class:`NonAstrophysicalInjector`
then turns a share of each batch's noise windows into non-astrophysical windows.

A real astrophysical signal is **coherent** across detectors: a shared source
gives a shared chirp mass and arrival times within the light-travel time. The
two *astrophysical* training classes are therefore:

  - signal + signal  (coherent injection)   -> class 1, both detectors supervised
  - noise  + noise   (pure noise)            -> class 0, neither supervised

Non-astrophysical pairs are a third class (2): the network estimates the
coherent-signal vs noise ratio with them as an auxiliary class (multinomial
density-ratio estimation). They take a share of the **noise** windows, never
of the signal budget, and are built from a separate *pool* of injections, one
per non-astrophysical window.

Two non-astrophysical flavours (split by ``signal_noise_share``):

  - signal + noise   : one detector carries a real chirp, the other is left as
                       pure noise. Mask = [1, 0] / [0, 1].
  - signal + signal' : each detector carries a chirp from a *different* event
                       (independent masses, independent arrival time). Mask =
                       [1, 1].

The thing that actually trips up a coincidence network is a non-astrophysical
pair whose chirps sit in the *real* ``delta_tc`` band — looking time-coincident while
being mass-inconsistent. So each detector's arrival time is re-drawn from a
**mixture**: inside the real ``delta_tc`` prior band with probability ``in_band_share``,
else anywhere in the analysis window, and the waveform is re-timed by a frequency-domain
phase shift ``exp(-2 pi i f dt)`` (same convention as
:meth:`sage.data.waveform.project.GravitationalWaveProjection.forward`). All
bounds are derived (real band from the ``delta_tc`` prior, window from the data
config) — nothing is hardcoded.

The per-detector **mask** (which detector carries a supervisable signal) is kept
separate from the **class** label (whether the pair is a coherent astrophysical
event): a decohered pair still has per-detector parameter targets while being
labelled "not a detection".

This is a TRAINING-ONLY augmentation; it must not be applied during validation.
Currently implemented for the two-detector case (general ``D`` supported).
"""

import math

import torch

NOISE_CLASS, SIGNAL_CLASS, NON_ASTROPHYSICAL_CLASS = 0, 1, 2
# Keep-out margin from the window edges for re-drawn arrival times.
EDGE_MARGIN_IN_S = 0.1


class NonAstrophysicalMasker:
    """Turn a pool of independent injections into non-astrophysical pairs.

    Parameters
    ----------
    delta_f : float
        Frequency resolution (Hz) of the pool's frequency-domain strain. The
        strain lives on a uniform real-FFT grid starting at DC, so the re-timing
        phase uses ``f[k] = k * delta_f`` over the data's own bin count — this is
        exactly the grid the projection applies its own delay on. Take it from
        the signal sampler (``signal_sampler.df``).
    delta_tc_bounds : tuple(float, float)
        The real ``delta_tc`` prior band ``(lo, hi)`` (derived from the parameter
        sampler). Re-drawn arrival times are weighted to favour this band.
    analysis_length_s : float
        Length of the analysis window in seconds (``data_cfg.sample_length_in_s``).
        The full re-timing span is ``[EDGE_MARGIN_IN_S, analysis_length_s - EDGE_MARGIN_IN_S]``.
    signal_noise_share : float
        Share of the pool made ``signal+noise`` (the rest ``signal+signal'``).
    in_band_share : float
        Per-detector probability that a re-drawn ``delta_tc`` lands in the real band
        rather than the full window. Higher -> more hard, time-coincident pairs.
    generator : torch.Generator
        Generator of every draw, on the pool's device.
    """

    def __init__(self, delta_f, delta_tc_bounds, analysis_length_s, signal_noise_share, in_band_share, generator):
        self.delta_f = float(delta_f)
        self.tc_lo, self.tc_hi = float(delta_tc_bounds[0]), float(delta_tc_bounds[1])
        self.full_lo = EDGE_MARGIN_IN_S
        self.full_hi = float(analysis_length_s) - EDGE_MARGIN_IN_S
        self.signal_noise_share = float(signal_noise_share)
        self.in_band_share = float(in_band_share)
        self.generator = generator

    def _sample_tc(self, n, device, dtype, g):
        """Mixture: ``in_band_share`` uniform in the real band, else uniform window."""
        in_band = torch.rand(n, device=device, generator=g) < self.in_band_share
        u = torch.rand(n, device=device, generator=g, dtype=dtype)
        band = self.tc_lo + (self.tc_hi - self.tc_lo) * u
        full = self.full_lo + (self.full_hi - self.full_lo) * u
        return torch.where(in_band, band, full)

    @torch.no_grad()
    def __call__(self, pool_data, pool_tc, pool_mc):
        """Build non-astrophysical injections from a pool of coherent signals.

        Parameters
        ----------
        pool_data : torch.Tensor, shape ``(N, D, F)`` complex
            Frequency-domain per-detector strain of ``N`` independent injections.
        pool_tc : torch.Tensor, shape ``(N, D)``
            Per-detector arrival times of the pool (physical seconds).
        pool_mc : torch.Tensor, shape ``(N, D)``
            Per-detector (standardised) chirp mass of the pool — identical across
            detectors per row on input (a coherent injection).

        Returns
        -------
        na_data : ``(N, D, F)``   non-astrophysical per-detector strain
        na_tc   : ``(N, D)``      re-drawn per-detector arrival times
        na_mc   : ``(N, D)``      per-detector (independent) chirp mass
        na_mask : ``(N, D)``      1 where a detector carries a supervisable signal
        """
        N, D, F = pool_data.shape
        device = pool_data.device
        if N == 0:
            empty_mask = torch.zeros(N, D, device=device, dtype=pool_tc.dtype)
            return pool_data, pool_tc.clone(), pool_mc.clone(), empty_mask

        g = self.generator
        # Real-FFT grid the strain lives on: f[k] = k * delta_f, DC-first, the
        # data's own bin count (matches the projection's delay grid exactly).
        f = self.delta_f * torch.arange(F, device=device, dtype=torch.float32)
        rows = torch.arange(N, device=device)

        # ── signal+signal': each detector a *different* event ──────────────────
        # detector d draws its channel from pool[(row + d) % N], so for D=2 the
        # first detector keeps its own event and the second takes the next row's
        # — independent masses and times. (RHS fully gathered before any write.)
        na_data = torch.empty_like(pool_data)
        src_tc = torch.empty(N, D, device=device, dtype=pool_tc.dtype)
        src_mc = torch.empty(N, D, device=device, dtype=pool_mc.dtype)
        for d in range(D):
            src = (rows + d) % N
            na_data[:, d] = pool_data[src, d]
            src_tc[:, d] = pool_tc[src, d]
            src_mc[:, d] = pool_mc[src, d]

        # ── re-time each detector to a weighted-random delta_tc (FD phase shift) ─────
        new_tc = torch.stack(
            [self._sample_tc(N, device, src_tc.dtype, g) for _ in range(D)], dim=1
        )                                                    # (N, D)
        dt = (new_tc - src_tc).to(f.dtype)                   # (N, D) seconds
        phase = torch.polar(
            torch.ones(N, D, F, device=device, dtype=f.dtype),
            (-2.0 * math.pi) * f.view(1, 1, F) * dt.unsqueeze(-1),
        )                                                    # (N, D, F) complex
        na_data = na_data * phase.to(na_data.dtype)

        # ── zero the circular-shift wraparound ────────────────────────────────
        # The FD phase shift is a *circular* shift in time: content pushed past
        # one window edge reappears at the other. We want a LINEAR shift — a
        # re-timed signal whose tail is simply cut off at the window edge, exactly
        # like a real time-domain search window catching a partial signal. We know
        # dt exactly, so we know which samples wrapped: a right shift (dt>0) wraps
        # the leading ``dt`` samples to the front, a left shift the trailing
        # ``|dt|`` to the back. Zero that band in the time domain. The merger lands
        # at ``new_tc``, always on the opposite side from the wrap, so it's never
        # touched.
        nsamples = 2 * (F - 1)
        td = torch.fft.irfft(na_data, n=nsamples, dim=-1)        # (N, D, nsamples)
        dt_samp = torch.round(dt * (nsamples * float(self.delta_f))).long()  # (N, D)
        idx = torch.arange(nsamples, device=device)
        ds = dt_samp.unsqueeze(-1)                               # (N, D, 1)
        keep = torch.where(ds > 0, idx >= ds, idx < nsamples + ds)  # (N, D, nsamples)
        na_data = torch.fft.rfft(td * keep, dim=-1).to(na_data.dtype)  # (N, D, F)

        na_tc = new_tc
        na_mc = src_mc
        na_mask = torch.ones(N, D, device=device, dtype=pool_tc.dtype)

        # ── split: a fixed fraction become signal+noise (drop all but one det) ─
        is_sn = torch.rand(N, device=device, generator=g) < self.signal_noise_share
        keep = torch.randint(0, D, (N,), device=device, generator=g)
        det_ids = torch.arange(D, device=device).view(1, D)
        keep_oh = det_ids == keep.view(N, 1)                 # (N, D) kept detector
        det_keep = torch.where(
            is_sn.view(N, 1), keep_oh, torch.ones_like(keep_oh)
        )                                                    # signal+signal' keeps all
        na_data = na_data * det_keep.unsqueeze(-1).to(na_data.dtype)
        na_mask = na_mask * det_keep.to(na_mask.dtype)

        return na_data, na_tc, na_mc, na_mask


class NonAstrophysicalInjector:
    """
    Turn a share of a batch's noise windows into non-astrophysical windows (class 2), without targets.

    Each noise window is chosen with probability ``noise_share`` and takes one pair; the pool draws a
    new block only when the pairs left from earlier blocks run out, so every pair drawn is used.

    Parameters
    ----------
    pool : torch.nn.Module
        Signal sampler with per-detector targets; each call draws one block.
    masker : NonAstrophysicalMasker
        Makes each block's signals non-astrophysical.
    noise_share : float
        Probability that a noise window receives a non-astrophysical pair.
    num_estimates : int
        Estimated parameters; the pool's per-detector arrival times follow them and the class.
    generator : torch.Generator
        Generator of the windows chosen, on the batch's device.
    """

    def __init__(self, pool, masker, noise_share, num_estimates, generator):
        self.pool = pool
        self.masker = masker
        self.noise_share = float(noise_share)
        self.num_estimates = num_estimates
        self.generator = generator
        self.pairs = None  # drawn, not yet used

    def take(self, count):
        """The next ``count`` pairs, drawing blocks from the pool while too few are left."""
        while (0 if self.pairs is None else len(self.pairs)) < count:
            data, targets = self.pool()
            first, detectors = self.num_estimates + 1, data.shape[1]
            arrival = targets[:, first : first + detectors]
            chirp_mass = targets[:, first + detectors : first + 2 * detectors]
            block = self.masker(data, arrival, chirp_mass)[0]
            self.pairs = block if self.pairs is None else torch.cat([self.pairs, block])
        pairs, self.pairs = self.pairs[:count], self.pairs[count:]
        return pairs

    def __call__(self, x, targets, signal_idx):
        """
        Add a pair to each chosen noise window of ``x`` and label it class 2; both are changed in place.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, F)``
            Batch, signals already injected.
        targets : torch.Tensor, shape ``(B, P + 1)``
            Its targets, the class last.
        signal_idx : torch.Tensor
            Positions of the signals.

        Returns
        -------
        x, targets : torch.Tensor
        """
        noise = torch.ones(len(x), dtype=torch.bool, device=x.device)
        noise[signal_idx] = False
        noise = noise.nonzero().squeeze(1)
        chosen = noise[torch.rand(len(noise), device=x.device, generator=self.generator) < self.noise_share]
        if len(chosen):
            x[chosen] += self.take(len(chosen)).to(x.dtype)
            targets[chosen, -1] = NON_ASTROPHYSICAL_CLASS
        return x, targets
