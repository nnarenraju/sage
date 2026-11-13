#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Torch helpers: in-place clamps with a margin, and JAX-style
``grad`` and ``value_and_grad``.

Filename        : torch.py
Description     : Short description of the file

Created on 2026-01-23 09:16:49

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import torch

from typing import Callable, Tuple, Union, Sequence


def nudge_backward_(foo: torch.Tensor, max_limit: float, nudge_factor=1e-6) -> None:
    """
    Clamp ``foo`` in place to at most ``max_limit - nudge_factor``.

    Parameters
    ----------
    foo : torch.Tensor
        Tensor, clamped in place.
    max_limit : float
        Upper limit.
    nudge_factor : float
        Margin kept below the limit.
    """
    foo.clamp_(max=max_limit - nudge_factor)


def nudge_forward_(foo: torch.Tensor, min_limit: float, nudge_factor=1e-6) -> None:
    """
    Clamp ``foo`` in place to at least ``min_limit + nudge_factor``.

    Parameters
    ----------
    foo : torch.Tensor
        Tensor, clamped in place.
    min_limit : float
        Lower limit.
    nudge_factor : float
        Margin kept above the limit.
    """
    foo.clamp_(min=min_limit + nudge_factor)


def torch_grad(func, args, argnums=0, create_graph=False):
    """
    Gradient of ``func`` with respect to ``args[argnums]``, as
    ``jax.grad``.

    Parameters
    ----------
    func : callable
        Function of ``args``.
    args : tuple of torch.Tensor
        Arguments of ``func``.
    argnums : int or tuple of int
        Arguments to differentiate with respect to.
    create_graph : bool
        Build the graph of the gradient, for higher derivatives.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        One gradient per entry of ``argnums``; a tensor for one.
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    args = list(args)

    # Enable gradients only for selected args
    for i in argnums:
        args[i] = args[i].requires_grad_(True)

    out = func(*args)

    # Handle scalar vs tensor output
    grad_outputs = None if out.ndim == 0 else torch.ones_like(out)

    grads = torch.autograd.grad(
        out,
        [args[i] for i in argnums],
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        allow_unused=True,
    )

    return grads[0] if len(grads) == 1 else grads


def torch_value_and_grad(
    fn: Callable,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    *,
    argnums: Union[int, Sequence[int]] = 0,
    create_graph: bool = False,
):
    """
    Value of ``fn`` and its gradient, as ``jax.value_and_grad``.

    Parameters
    ----------
    fn : callable
        Function of ``inputs``.
    inputs : torch.Tensor or tuple of torch.Tensor
        Arguments of ``fn``.
    argnums : int or sequence of int
        Arguments to differentiate with respect to; the first by
        default.
    create_graph : bool
        Build the graph of the gradient, for higher derivatives.

    Returns
    -------
    value : torch.Tensor
        ``fn(*inputs)``.
    grads : torch.Tensor or tuple of torch.Tensor
        One gradient per entry of ``argnums``; a tensor for one.
    """

    # Normalize inputs to tuple
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    # Normalize argnums
    if isinstance(argnums, int):
        argnums = (argnums,)

    # Ensure requires_grad only for requested args
    inputs = list(inputs)
    for i in argnums:
        if not inputs[i].requires_grad:
            inputs[i].requires_grad_(True)

    # Forward pass
    value = fn(*inputs)

    # Backward target
    grad_outputs = None if value.ndim == 0 else torch.ones_like(value)

    # Compute gradients only w.r.t. selected inputs
    grads = torch.autograd.grad(
        value,
        [inputs[i] for i in argnums],
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )

    # Match JAX return shape
    if len(grads) == 1:
        grads = grads[0]

    return value, grads
