#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : normalise.py
Description   : Per-sample, per-detector min-max normalisation of whitened strain.

Faithful port of the Sept-2024 paper run's ``transforms.Normalise(ignore_factors
=True)`` (git d4cc814, data/transforms.py:268), which sat in the transform stack
BETWEEN whitening and multirate sampling (configs.py stage2):

    stage1: Whiten(remove_corrupted=True)      # -> 12 s valid
    stage2: Normalise(ignore_factors=True)     # <-- THIS
            MultirateSampling()

The old op, per detector channel, per sample:

    y -> (y - min(y)) / (max(y) - min(y))      # -> [0, 1]
    y ->  y - mean(y)                          # centre -> ~[-0.5, 0.5]

It bounds every sample -- including glitch segments whose whitened strain
spikes to O(100-1000) -- to a fixed range before the multirate decimation runs,
which improves numerical conditioning of the downstream filtering/optimisation.

Note: this is a per-detector affine map, so a per-detector InstanceNorm applied
later is (to first order) invariant to it; its effect is (a) numerical stability
before multirate and (b) it is NOT undone by GroupNorm(1, D), which normalises
across detectors.
"""

import torch

from sage.core.pipeline import GWBatch


class MinMaxNormalise(torch.nn.Module):
    """Per-(sample, detector) min-max-to-[0,1] then mean-centre, over the time axis.

    Accepts a :class:`GWBatch` (state-tracked path) or a raw ``(B, D, T)`` tensor
    (legacy path).  Operates only on time-domain data; the processing state is
    passed through unchanged.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def _apply_td(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) real. Reduce over the time axis, per (sample, detector).
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        y = (x - mn) / (mx - mn + self.eps)          # -> [0, 1]
        y = y - y.mean(dim=-1, keepdim=True)          # centre -> ~[-0.5, 0.5]
        return y

    @torch.no_grad()
    def forward(self, input):
        if isinstance(input, GWBatch):
            return GWBatch(
                self._apply_td(input.data),
                input.state,
                input.freqs,
                input.coarse_indices,
            )
        return self._apply_td(input)
