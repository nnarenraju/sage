#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pipeline building blocks: the ordered preprocessing chain
(``Preprocessor``) and a per-sample module choice (``TorchChoice``)."""

import torch
import torch.nn as nn

from typing import List

from sage.core.pipeline import GWBatch, PipelineError, ProcessingState


class Preprocessor(nn.Module):
    """
    Ordered preprocessing chain for gravitational-wave batches.

    Each step declares the state it takes and gives through a static
    ``next_state(state)``; the chain is checked at construction. Input
    and output are ``GWBatch`` objects.

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

    Examples
    --------
    >>> Preprocessor([FiducialWhitening(asds, window, device),
    ...               MultirateSampler(bins, window, device, dtype)],
    ...              ProcessingState(Grid.FD_UNIFORM))
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
    One module per sample, drawn with the given probabilities.

    Parameters
    ----------
    modules : list of nn.Module
        Candidate modules.
    probabilities : list of float
        Probability of each module; normalised to sum to 1.
    """

    def __init__(self, modules: List[nn.Module], probabilities: List[float]):
        super().__init__()
        assert len(modules) == len(probabilities)
        self.modules_list = nn.ModuleList(modules)
        probs = torch.tensor(probabilities, dtype=torch.float32)
        self.register_buffer("probs", probs / probs.sum())

    def forward(self, x, generator=None):
        """
        Apply a randomly chosen module to each sample.

        Parameters
        ----------
        x : torch.Tensor
            Batch, samples along the first axis.
        generator : torch.Generator, optional
            Random generator for the choice.

        Returns
        -------
        torch.Tensor
            Batch of the same shape.
        """
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
