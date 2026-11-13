#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-astrophysical (class 2) detector pairs for training: chirps that
look time-coincident but are inconsistent, as signal + noise or signal
+ another event's signal, added to a share of the noise windows.
"""

import math

import torch

NOISE_CLASS, SIGNAL_CLASS, NON_ASTROPHYSICAL_CLASS = 0, 1, 2
# Keep-out margin from the window edges for re-drawn arrival times.
EDGE_MARGIN_IN_S = 0.1


class NonAstrophysicalMasker:
    """
    Turn a pool of independent injections into non-astrophysical pairs.

    Each detector's arrival time is redrawn, inside the real
    ``delta_tc`` band with probability ``in_band_share`` and anywhere in
    the window otherwise, and applied by a frequency-domain phase shift.

    Parameters
    ----------
    delta_f : float
        Frequency resolution (Hz) of the pool's strain, on a real-FFT
        grid from DC (``signal_sampler.df``).
    delta_tc_bounds : tuple of float
        The real ``delta_tc`` prior band ``(lo, hi)``.
    analysis_length_s : float
        Analysis window length (s); arrival times stay
        ``EDGE_MARGIN_IN_S`` from its edges.
    signal_noise_share : float
        Share of the pairs made signal + noise; the rest are signal +
        signal of different events.
    in_band_share : float
        Per-detector probability that an arrival time is drawn in the
        real band.
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
        """
        Arrival times, in the real band or over the whole window.

        A time is uniform in the real band with probability
        ``in_band_share``, else uniform over the window.

        Parameters
        ----------
        n : int
            Number of times.
        device : torch.device
            Device.
        dtype : torch.dtype
            Dtype of the times.
        g : torch.Generator
            Generator.

        Returns
        -------
        torch.Tensor, shape ``(n,)``
            Arrival times (s).
        """
        in_band = torch.rand(n, device=device, generator=g) < self.in_band_share
        u = torch.rand(n, device=device, generator=g, dtype=dtype)
        band = self.tc_lo + (self.tc_hi - self.tc_lo) * u
        full = self.full_lo + (self.full_hi - self.full_lo) * u
        return torch.where(in_band, band, full)

    @torch.no_grad()
    def __call__(self, pool_data, pool_tc, pool_mc):
        """
        Build non-astrophysical pairs from a pool of coherent signals.

        Detector ``d`` of row ``i`` takes the signal of row
        ``(i + d) % N``, re-timed to a drawn arrival time, with the
        samples wrapped by the circular shift zeroed. A share
        ``signal_noise_share`` of rows keep one detector only.

        Parameters
        ----------
        pool_data : torch.Tensor, shape ``(N, D, F)``
            Frequency-domain strain of ``N`` independent injections.
        pool_tc : torch.Tensor, shape ``(N, D)``
            Per-detector arrival times (s).
        pool_mc : torch.Tensor, shape ``(N, D)``
            Per-detector standardised chirp mass, equal across a row.

        Returns
        -------
        na_data : torch.Tensor, shape ``(N, D, F)``
            Non-astrophysical strain.
        na_tc : torch.Tensor, shape ``(N, D)``
            Redrawn arrival times.
        na_mc : torch.Tensor, shape ``(N, D)``
            Chirp mass of each detector's event.
        na_mask : torch.Tensor, shape ``(N, D)``
            1 where a detector carries a signal.
        """
        N, D, F = pool_data.shape
        device = pool_data.device
        if N == 0:
            empty_mask = torch.zeros(N, D, device=device, dtype=pool_tc.dtype)
            return pool_data, pool_tc.clone(), pool_mc.clone(), empty_mask

        g = self.generator
        # Real-FFT grid of the strain, f[k] = k * delta_f, as in the
        # projection
        f = self.delta_f * torch.arange(F, device=device, dtype=torch.float32)
        rows = torch.arange(N, device=device)

        # Detector d takes row (row + d) % N: a different event per
        # detector
        na_data = torch.empty_like(pool_data)
        src_tc = torch.empty(N, D, device=device, dtype=pool_tc.dtype)
        src_mc = torch.empty(N, D, device=device, dtype=pool_mc.dtype)
        for d in range(D):
            src = (rows + d) % N
            na_data[:, d] = pool_data[src, d]
            src_tc[:, d] = pool_tc[src, d]
            src_mc[:, d] = pool_mc[src, d]

        # Re-time each detector to a drawn arrival time (FD phase shift)
        new_tc = torch.stack(
            [self._sample_tc(N, device, src_tc.dtype, g) for _ in range(D)], dim=1
        )                                                    # (N, D)
        dt = (new_tc - src_tc).to(f.dtype)                   # (N, D) seconds
        phase = torch.polar(
            torch.ones(N, D, F, device=device, dtype=f.dtype),
            (-2.0 * math.pi) * f.view(1, 1, F) * dt.unsqueeze(-1),
        )                                                    # (N, D, F) complex
        na_data = na_data * phase.to(na_data.dtype)

        # The phase shift is circular in time; zeroing the wrapped
        # samples (the first dt for dt > 0, the last |dt| otherwise)
        # makes it a linear shift. The merger is never zeroed.
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

        # signal + noise rows keep one random detector
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
    Turn a share of a batch's noise windows into non-astrophysical ones.

    Each noise window is chosen with probability ``noise_share`` and
    takes one pair; the pool draws a new block only when the pairs left
    run out. The windows get class 2 and no parameter targets.

    Parameters
    ----------
    pool : torch.nn.Module
        Signal sampler with per-detector targets; each call draws one
        block.
    masker : NonAstrophysicalMasker
        Makes each block's signals non-astrophysical.
    noise_share : float
        Probability that a noise window receives a pair.
    num_estimates : int
        Number of estimated parameters; the pool's targets hold them,
        the class, then per-detector arrival times and chirp masses.
    generator : torch.Generator
        Generator of the window choice, on the batch's device.
    """

    def __init__(self, pool, masker, noise_share, num_estimates, generator):
        self.pool = pool
        self.masker = masker
        self.noise_share = float(noise_share)
        self.num_estimates = num_estimates
        self.generator = generator
        self.pairs = None  # drawn, not yet used

    def take(self, count):
        """
        The next ``count`` pairs, drawing pool blocks as needed.

        Parameters
        ----------
        count : int
            Number of pairs.

        Returns
        -------
        torch.Tensor, shape ``(count, D, F)``
            Non-astrophysical strain.
        """
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
        Add a pair to each chosen noise window and label it class 2.

        ``x`` and ``targets`` are changed in place.

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
            The changed batch and targets.
        """
        noise = torch.ones(len(x), dtype=torch.bool, device=x.device)
        noise[signal_idx] = False
        noise = noise.nonzero().squeeze(1)
        chosen = noise[torch.rand(len(noise), device=x.device, generator=self.generator) < self.noise_share]
        if len(chosen):
            x[chosen] += self.take(len(chosen)).to(x.dtype)
            targets[chosen, -1] = NON_ASTROPHYSICAL_CLASS
        return x, targets
