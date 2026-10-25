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

from sage.core.pipeline import GWBatch, PipelineError, ProcessingState


class MinMaxNormalise(torch.nn.Module):
    """Per-(sample, detector) min-max-to-[0,1] then mean-centre, over the time axis.

    Takes time-domain batches only; the processing state is passed through
    unchanged. ``EPS`` keeps a constant channel at zero instead of 0/0.
    """

    EPS = 1e-12

    @staticmethod
    def next_state(state: ProcessingState) -> ProcessingState:
        """
        State after min-max normalisation: unchanged.

        Parameters
        ----------
        state : ProcessingState
            State of the incoming batch.

        Returns
        -------
        ProcessingState
            The same state.

        Raises
        ------
        PipelineError
            If the batch is not time-domain data.
        """
        if not state.is_td():
            raise PipelineError(f"Min-max normalisation needs time-domain data, got {state}.")
        return state

    def _apply_td(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T) real. Reduce over the time axis, per (sample, detector).
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        y = (x - mn) / (mx - mn + self.EPS)          # -> [0, 1]
        y = y - y.mean(dim=-1, keepdim=True)          # centre -> ~[-0.5, 0.5]
        return y

    @torch.no_grad()
    def forward(self, batch: GWBatch) -> GWBatch:
        """
        Normalise each (sample, detector) channel of a time-domain batch.

        Parameters
        ----------
        batch : GWBatch
            Time-domain strain ``(B, D, T)``.

        Returns
        -------
        GWBatch
            Normalised strain, same state.
        """
        new_state = self.next_state(batch.state)
        return GWBatch(self._apply_td(batch.data), new_state)
