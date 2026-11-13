#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Per-sample, per-detector min-max normalisation of time-domain strain,
as a pipeline step.

Filename      : normalise.py
Description   : Per-sample, per-detector min-max normalisation of whitened strain.
"""

import torch

from sage.core.pipeline import GWBatch, PipelineError, ProcessingState


class MinMaxNormalise(torch.nn.Module):
    """
    Min-max scaling to [0, 1], then mean-centring, over the time axis.

    Per sample and detector, on time-domain batches; the processing
    state is unchanged. ``EPS`` maps a constant channel to zero.
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
        """
        Normalise each (sample, detector) channel over the time axis.

        Parameters
        ----------
        x : torch.Tensor, shape ``(B, D, T)``
            Real strain.

        Returns
        -------
        torch.Tensor, shape ``(B, D, T)``
            Scaled to [0, 1], then centred on zero mean.
        """
        mn = x.amin(dim=-1, keepdim=True)
        mx = x.amax(dim=-1, keepdim=True)
        y = (x - mn) / (mx - mn + self.EPS)          # -> [0, 1]
        y = y - y.mean(dim=-1, keepdim=True)          # zero mean
        return y

    @torch.no_grad()
    def forward(self, batch: GWBatch) -> GWBatch:
        """
        Normalise each channel of a time-domain batch.

        Parameters
        ----------
        batch : GWBatch
            Time-domain strain ``(B, D, T)``.

        Returns
        -------
        GWBatch
            Normalised strain, same state.

        Raises
        ------
        PipelineError
            If the batch is not time-domain data.
        """
        new_state = self.next_state(batch.state)
        return GWBatch(self._apply_td(batch.data), new_state)
