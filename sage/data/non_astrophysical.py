#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-astrophysical (decoherent) two-detector sample generation for training the
multi-detector consistency heads to reject incoherent coincidences.

A real astrophysical signal is **coherent** across detectors (shared source ->
shared chirp mass, arrival times within the light-travel time). The two
*astrophysical* training classes are therefore:

  - signal + signal  (coherent injection)   -> class 1, both detectors supervised
  - noise  + noise   (pure noise)            -> class 0, neither supervised

To teach the network that coincidence alone is not detection, a small fraction
of the coherent injections are *decohered* (a "time slide" — shifting one
detector relative to the other) into the two **non-astrophysical** classes:

  - signal + noise   : the signal is removed from one detector (the slide lands
                       on noise). Mask = [1, 0] / [0, 1]; class 0.
  - signal + signal' : one detector's signal is replaced by a *different*
                       injection's (the slide lands on another signal). Mask =
                       [1, 1] (each detector toward its own truth); class 0.

The per-detector **mask** (which detector carries a supervisable signal) is kept
separate from the **class** label (whether the pair is a coherent astrophysical
event) — a decohered pair can still have per-detector parameter targets while
being labelled "not a detection".

This is a TRAINING-ONLY augmentation; it must not be applied during validation.
Currently implemented for the two-detector case.
"""

import torch


class NonAstrophysicalMasker:
    """Decohere a fraction of coherent injections into non-astrophysical pairs.

    Parameters
    ----------
    p_non_astro : float
        Probability that a given injected signal is decohered into a
        non-astrophysical pair. ``0`` disables it (all injections stay coherent).
    p_signal_noise : float
        Of the decohered injections, the fraction made ``signal+noise`` (the rest
        are ``signal+signal'``). Default ``0.5``.
    seed : int or None
        Optional RNG seed (a device generator is created lazily on first call).
    """

    def __init__(self, p_non_astro: float = 0.0, p_signal_noise: float = 0.5, seed=None):
        self.p = float(p_non_astro)
        self.p_sn = float(p_signal_noise)
        self._seed = seed
        self._gen = None

    def _generator(self, device):
        if self._seed is not None and self._gen is None:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(int(self._seed))
        return self._gen

    @torch.no_grad()
    def __call__(self, signal_data, per_det_tc):
        """Decohere a random subset of the signal batch.

        Parameters
        ----------
        signal_data : torch.Tensor, shape ``(S, 2, F)``
            Per-detector signal injections (coherent on input).
        per_det_tc : torch.Tensor, shape ``(S, 2)``
            Per-detector arrival-time targets (seconds).

        Returns
        -------
        signal_data : ``(S, 2, F)``   possibly with a detector zeroed/replaced
        per_det_tc  : ``(S, 2)``      updated for replaced detectors
        signal_mask : ``(S, 2)``      1 where a detector carries a supervisable signal
        is_coherent : ``(S,)``        1 for coherent injections, 0 for decohered
        """
        S, D = signal_data.shape[0], signal_data.shape[1]
        device = signal_data.device
        signal_mask = torch.ones(S, D, device=device, dtype=per_det_tc.dtype)
        is_coherent = torch.ones(S, device=device, dtype=per_det_tc.dtype)

        if self.p <= 0.0 or D != 2:
            return signal_data, per_det_tc, signal_mask, is_coherent

        g = self._generator(device)
        signal_data = signal_data.clone()
        per_det_tc = per_det_tc.clone()

        non_astro = torch.rand(S, device=device, generator=g) < self.p
        is_sn = non_astro & (torch.rand(S, device=device, generator=g) < self.p_sn)
        is_ss = non_astro & ~is_sn
        det = (torch.rand(S, device=device, generator=g) < 0.5).long()  # decohered ifo
        perm = torch.randperm(S, device=device, generator=g)
        rows = torch.arange(S, device=device)

        # signal + noise: drop the decohered detector's signal (now pure noise).
        sn, sn_d = rows[is_sn], det[is_sn]
        if sn.numel():
            signal_data[sn, sn_d] = 0
            signal_mask[sn, sn_d] = 0.0

        # signal + signal': replace the decohered detector with another injection
        # (RHS is fully gathered before the in-place assignment, so no hazard).
        ss, ss_d = rows[is_ss], det[is_ss]
        if ss.numel():
            signal_data[ss, ss_d] = signal_data[perm[ss], ss_d]
            per_det_tc[ss, ss_d] = per_det_tc[perm[ss], ss_d]

        is_coherent[non_astro] = 0.0
        return signal_data, per_det_tc, signal_mask, is_coherent
