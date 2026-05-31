#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sage pipeline building blocks.

Preprocessor
    Sequential chain of preprocessing modules (whitening, multirate, etc.).
    Passes data through each module in order.  Modules may accept and return
    either raw tensors (legacy) or :class:`~sage.core.pipeline.GWBatch`
    objects — ``nn.Sequential`` is type-agnostic, so the chain works in both
    cases transparently.

TorchChoice
    Probabilistic per-sample module selection.  Useful for augmentation or
    domain randomisation.
"""

# Packages
import torch
import torch.nn as nn

from typing import List


class Preprocessor(nn.Module):
    """
    Sequential preprocessing pipeline for gravitational-wave data.

    Chains an ordered list of ``nn.Module`` transforms.  Each module receives
    the output of the previous one.  A typical pipeline for TD training is::

        Preprocessor([FiducialWhitening(), MultirateSampler(...)])

    and for FD_COARSE (worst-case multibanding)::

        Preprocessor([FiducialWhitening()])

    The pipeline is grid-aware when modules return
    :class:`~sage.core.pipeline.GWBatch` objects (which is the default for
    updated modules).  The training loop is responsible for wrapping the
    combined signal+noise tensor in a GWBatch before calling this module,
    and for extracting ``batch.to_network_input()`` afterward.

    Parameters
    ----------
    modules : list of nn.Module
        Ordered preprocessing steps.
    """

    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        x : torch.Tensor or GWBatch
            Input data.  Shape and type depend on the first module.

        Returns
        -------
        torch.Tensor or GWBatch
            Preprocessed data.  Type matches whatever the last module returns.
        """
        return self.seq(x)


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
