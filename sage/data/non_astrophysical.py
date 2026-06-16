#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-astrophysical (decoherent) two-detector sample generation for training the
multi-detector consistency heads to reject incoherent coincidences.

A real astrophysical signal is **coherent** across detectors: a shared source
gives a shared chirp mass and arrival times within the light-travel time. The
two *astrophysical* training classes are therefore:

  - signal + signal  (coherent injection)   -> class 1, both detectors supervised
  - noise  + noise   (pure noise)            -> class 0, neither supervised

To teach the network that coincidence alone is not detection, a small fraction
of the batch is replaced by **non-astrophysical** pairs. Crucially these are
*background* (class 0): they must eat into the **noise** budget, never the
signal budget, or they would unbalance the classes. They are therefore built
from a separate *pool* of extra injections (``extra_batch`` on the signal
sampler) and dropped into noise slots, not carved out of the coherent signals.

Two non-astrophysical flavours (split 50/50, fixed):

  - signal + noise   : one detector carries a real chirp, the other is left as
                       pure noise. Mask = [1, 0] / [0, 1]; class 0.
  - signal + signal' : each detector carries a chirp from a *different* event
                       (independent masses, independent arrival time). Mask =
                       [1, 1] (each detector toward its own truth); class 0.

The thing that actually trips up a coincidence network is a non-astrophysical
pair whose chirps sit in the *real* ``tc`` band — looking time-coincident while
being mass-inconsistent. So each detector's arrival time is re-drawn from a
**mixture** that favours the real ``tc`` prior band and otherwise spreads across
the whole analysis window, and the waveform is re-timed by a frequency-domain
phase shift ``exp(-2 pi i f dt)`` (same convention as
:meth:`sage.data.waveform.project.GravitationalWaveProjection.forward`). All
bounds are derived (real band from the ``tc`` prior, window from the data
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


class NonAstrophysicalMasker:
    """Turn a pool of independent injections into non-astrophysical class-0 pairs.

    Parameters
    ----------
    freqs : torch.Tensor, shape ``(F,)``
        The (real, positive) frequency grid the pool's frequency-domain strain
        lives on — used for the re-timing phase shift. Take it from the signal
        sampler (``signal_sampler.f[0]``).
    tc_bounds : tuple(float, float)
        The real ``tc`` prior band ``(lo, hi)`` (derived from the parameter
        sampler). Re-drawn arrival times are weighted to favour this band.
    analysis_length_s : float
        Length of the analysis window in seconds (``data_cfg.sample_length_in_s``).
        The full re-timing span is ``[edge_margin_s, analysis_length_s - edge_margin_s]``.
    p_signal_noise : float
        Fraction of the pool made ``signal+noise`` (the rest ``signal+signal'``).
        Fixed at ``0.5`` by design; exposed only for testing.
    w_in_band : float
        Per-detector probability that a re-drawn ``tc`` lands in the real band
        rather than the full window. Higher -> more hard, time-coincident pairs.
    edge_margin_s : float
        Keep-out margin (seconds) from the window edges for re-drawn ``tc``.
    seed : int or None
        Optional RNG seed (a device generator is created lazily on first call).
    """

    def __init__(
        self,
        freqs,
        tc_bounds,
        analysis_length_s,
        p_signal_noise: float = 0.5,
        w_in_band: float = 0.5,
        edge_margin_s: float = 0.1,
        seed=None,
    ):
        self._freqs = torch.as_tensor(freqs).reshape(-1).float()
        self._freqs_dev = None  # cached device copy
        self.tc_lo, self.tc_hi = float(tc_bounds[0]), float(tc_bounds[1])
        self.full_lo = float(edge_margin_s)
        self.full_hi = float(analysis_length_s) - float(edge_margin_s)
        self.p_sn = float(p_signal_noise)
        self.w_in_band = float(w_in_band)
        self._seed = seed
        self._gen = None

    def _generator(self, device):
        if self._seed is not None and self._gen is None:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(int(self._seed))
        return self._gen

    def _freqs_on(self, device):
        if self._freqs_dev is None or self._freqs_dev.device != device:
            self._freqs_dev = self._freqs.to(device)
        return self._freqs_dev

    def _sample_tc(self, n, device, dtype, g):
        """Mixture: ``w_in_band`` uniform in the real band, else uniform window."""
        in_band = torch.rand(n, device=device, generator=g) < self.w_in_band
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
        na_data : ``(N, D, F)``   non-astrophysical per-detector strain (class 0)
        na_tc   : ``(N, D)``      re-drawn per-detector arrival times
        na_mc   : ``(N, D)``      per-detector (independent) chirp mass
        na_mask : ``(N, D)``      1 where a detector carries a supervisable signal
        """
        N, D, F = pool_data.shape
        device = pool_data.device
        if N == 0:
            empty_mask = torch.zeros(N, D, device=device, dtype=pool_tc.dtype)
            return pool_data, pool_tc.clone(), pool_mc.clone(), empty_mask

        g = self._generator(device)
        f = self._freqs_on(device)
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

        # ── re-time each detector to a weighted-random tc (FD phase shift) ─────
        new_tc = torch.stack(
            [self._sample_tc(N, device, src_tc.dtype, g) for _ in range(D)], dim=1
        )                                                    # (N, D)
        dt = (new_tc - src_tc).to(f.dtype)                   # (N, D) seconds
        phase = torch.polar(
            torch.ones(N, D, F, device=device, dtype=f.dtype),
            (-2.0 * math.pi) * f.view(1, 1, F) * dt.unsqueeze(-1),
        )                                                    # (N, D, F) complex
        na_data = na_data * phase.to(na_data.dtype)

        na_tc = new_tc
        na_mc = src_mc
        na_mask = torch.ones(N, D, device=device, dtype=pool_tc.dtype)

        # ── split: a fixed fraction become signal+noise (drop all but one det) ─
        is_sn = torch.rand(N, device=device, generator=g) < self.p_sn
        keep = torch.randint(0, D, (N,), device=device, generator=g)
        det_ids = torch.arange(D, device=device).view(1, D)
        keep_oh = det_ids == keep.view(N, 1)                 # (N, D) kept detector
        det_keep = torch.where(
            is_sn.view(N, 1), keep_oh, torch.ones_like(keep_oh)
        )                                                    # signal+signal' keeps all
        na_data = na_data * det_keep.unsqueeze(-1).to(na_data.dtype)
        na_mask = na_mask * det_keep.to(na_mask.dtype)

        return na_data, na_tc, na_mc, na_mask
