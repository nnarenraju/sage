#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sage pipeline building blocks.

Preprocessor
    Ordered chain of preprocessing steps (whitening, min-max normalisation,
    multirate sampling) on :class:`~sage.core.pipeline.GWBatch` objects. The
    order is checked when the chain is built.

TorchChoice
    Probabilistic per-sample module selection.  Useful for augmentation or
    domain randomisation.
"""

# Packages
import torch
import torch.nn as nn

from typing import List

from sage.core.pipeline import GWBatch, PipelineError, ProcessingState


class Preprocessor(nn.Module):
    """
    Ordered preprocessing chain for gravitational-wave batches.

    Each step declares the state it takes and gives through a static
    ``next_state(state)``. The chain is walked once at construction, so a step
    in an impossible position raises before any data flows. For example::

        Preprocessor(
            [FiducialWhitening(asds, window, device),
             MultirateSampler(bins, window, device, dtype)],
            ProcessingState(Grid.FD_UNIFORM),
        )

    The caller wraps the input in a GWBatch and takes
    ``batch.to_network_input()`` from the result.

    Parameters
    ----------
    modules : list of nn.Module
        Ordered preprocessing steps, each with a static ``next_state``.
    input_state : ProcessingState
        State of the batches the chain receives.

    Attributes
    ----------
    input_state, output_state : ProcessingState
        States before the first and after the last step.

    Raises
    ------
    PipelineError
        If a step cannot take the state the step before it gives.
    TypeError
        If a module does not declare ``next_state``.
    """

    def __init__(self, modules: List[nn.Module], input_state: ProcessingState):
        super().__init__()
        state = input_state
        for index, module in enumerate(modules):
            where = f"preprocess[{index}] ({type(module).__name__})"
            if not hasattr(module, "next_state"):
                raise TypeError(f"{where} does not declare next_state")
            try:
                state = module.next_state(state)
            except PipelineError as error:
                raise PipelineError(f"{where}: {error}") from None
        self.input_state = input_state
        self.output_state = state
        self.seq = nn.Sequential(*modules)

    def forward(self, batch: GWBatch) -> GWBatch:
        """
        Run the chain.

        Parameters
        ----------
        batch : GWBatch
            Batch in ``input_state``.

        Returns
        -------
        GWBatch
            Batch in ``output_state``.

        Raises
        ------
        PipelineError
            If the batch is not in ``input_state``.
        """
        if batch.state != self.input_state:
            raise PipelineError(
                f"preprocessor built for {self.input_state}, got a batch in {batch.state}"
            )
        return self.seq(batch)


class TorchChoice(nn.Module):
    """
    Choose one module per sample according to provided probabilities.

    Useful for data augmentation that should apply different transforms
    to different samples within a batch.

    Parameters
    ----------
    modules : list of nn.Module
        Candidate modules.
    probabilities : list of float
        Probability of selecting each module (automatically normalised).
    """

    def __init__(self, modules: List[nn.Module], probabilities: List[float]):
        super().__init__()
        assert len(modules) == len(probabilities)
        self.modules_list = nn.ModuleList(modules)
        probs = torch.tensor(probabilities, dtype=torch.float32)
        self.register_buffer("probs", probs / probs.sum())

    def forward(self, x, generator=None):
        B = x.shape[0]
        device = x.device
        probs = self.probs.to(device)
        dist = torch.distributions.Categorical(probs)
        choices = dist.sample((B,), generator=generator)

        output = torch.empty_like(x)
        for idx, module in enumerate(self.modules_list):
            idxs = torch.nonzero(choices == idx, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                continue
            selected = x.index_select(0, idxs)
            processed = module(selected)
            output.index_copy_(0, idxs, processed)

        return output
