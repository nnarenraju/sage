#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte-Carlo dropout inference utilities.

At test time, keeping the dropout layers stochastic and averaging several
forward passes turns the network into an approximate Bayesian model: the
*spread* across passes is an estimate of **epistemic** uncertainty. With the
heteroscedastic head this complements the predicted (aleatoric) sigma —
together they give a total predictive uncertainty.

These helpers are model-agnostic: they work for any model whose forward returns
a tensor, a tuple of tensors, or a NamedTuple of tensors. They are provided for
downstream use and are not wired into the training/validation loops.
"""

import torch
import torch.nn as nn

_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Put the model in eval mode but re-enable *only* the dropout layers.

    BatchNorm / InstanceNorm and everything else stay in eval mode (using their
    running statistics); the dropout layers become stochastic again so repeated
    forward passes differ.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
    return model


def _stack_reduce(samples):
    """Mean/std over a list of identically-structured outputs."""
    first = samples[0]
    if torch.is_tensor(first):
        s = torch.stack(samples, dim=0)
        return s.mean(0), s.std(0)
    # tuple / NamedTuple of tensors
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
    """Run ``n_samples`` stochastic-dropout forward passes and reduce.

    Parameters
    ----------
    model : nn.Module
        A model containing dropout layers (e.g. built with ``dropout > 0``).
    x : input accepted by ``model.forward``
    n_samples : int
        Number of stochastic passes.

    Returns
    -------
    mean, std
        The per-element mean and standard deviation across passes, with the same
        structure as the model output (tensor / tuple / NamedTuple). The model
        is left in MC-dropout mode; call ``model.eval()`` to restore.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    enable_mc_dropout(model)
    samples = [model(x) for _ in range(n_samples)]
    return _stack_reduce(samples)
