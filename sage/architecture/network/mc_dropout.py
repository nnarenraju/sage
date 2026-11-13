#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte-Carlo dropout inference: the spread over stochastic passes.

Works for any model whose forward returns a tensor, a tuple of
tensors or a NamedTuple of tensors. Not used by the training or
validation loops.
"""

import torch
import torch.nn as nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """
    Eval mode for the model, train mode for its dropout layers.

    Parameters
    ----------
    model : nn.Module
        Model to switch, in place.

    Returns
    -------
    nn.Module
        The same model.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
    return model


def _stack_reduce(samples):
    """
    Mean and standard deviation over identically structured outputs.

    Parameters
    ----------
    samples : list
        Outputs of repeated passes: tensors, tuples or NamedTuples.

    Returns
    -------
    mean, std
        Elementwise mean and standard deviation, in the structure of
        the outputs.
    """
    first = samples[0]
    if torch.is_tensor(first):
        s = torch.stack(samples, dim=0)
        return s.mean(0), s.std(0)
    # Tuple or NamedTuple of tensors
    means, stds = [], []
    for i in range(len(first)):
        s = torch.stack([out[i] for out in samples], dim=0)
        means.append(s.mean(0))
        stds.append(s.std(0))
    cls = type(first)
    try:                       # NamedTuple
        return cls(*means), cls(*stds)
    except TypeError:          # plain tuple
        return tuple(means), tuple(stds)


@torch.no_grad()
def mc_predict(model: nn.Module, x, n_samples: int = 30):
    """
    Mean and spread of ``n_samples`` stochastic-dropout forward passes.

    The model is left in MC-dropout mode; ``model.eval()`` restores it.

    Parameters
    ----------
    model : nn.Module
        Model with dropout layers, e.g. built with ``dropout > 0``.
    x : object
        Input accepted by ``model.forward``.
    n_samples : int
        Number of stochastic passes.

    Returns
    -------
    mean, std
        Elementwise mean and standard deviation over the passes, in the
        structure of the model output.

    Raises
    ------
    ValueError
        If ``n_samples`` is below 1.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    enable_mc_dropout(model)
    samples = [model(x) for _ in range(n_samples)]
    return _stack_reduce(samples)
