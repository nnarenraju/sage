#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : utils.py
Description     : Short description of the file

Created on 2025-11-06 17:39:59

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import torch
import numpy as np

from typing import Callable, Tuple, Union, Sequence


def to_sequence(x):
    """Wrap scalars into a tuple so everything becomes iterable."""
    if x is None:
        return x
    if isinstance(x, (str, bytes)):
        return (x,)  # single string as a tuple
    if isinstance(x, (int, float)):
        return (x,)
    if isinstance(x, (bool)):
        return (x,)
    return tuple(x)


def ensure_1d(x):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1D")
    return x


def torch_grad(func, args, argnums=0, create_graph=False):
    """
    PyTorch equivalent of jax.grad.

    Computes gradient of func w.r.t. args[argnums].

    Args:
        func: callable
        args: tuple of arguments passed to func
        argnums: int or tuple of ints (default: 0)
        create_graph: whether to construct graph for higher-order grads

    Returns:
        Gradient(s) corresponding to argnums
        - Tensor if single argnum
        - Tuple of tensors if multiple argnums
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
    Torch equivalent of jax.value_and_grad(fn, argnums).

    Call style:
        value, grads = torch_value_and_grad(fn, (arg1, arg2, arg3))
        value, grads = torch_value_and_grad(fn, (arg1, arg2, arg3), argnums=(0, 1))

    Default:
        Differentiates w.r.t. argnums=0 (first argument only),
        matching JAX behavior.

    Args:
    -----
        fn: callable
        inputs: Tensor or tuple of Tensors
        argnums: int or tuple of ints specifying which arguments to differentiate
        create_graph: whether to construct higher-order graph

    Returns:
    --------
        value: fn(*inputs)
        grads: gradient(s) w.r.t. argnums
               - Tensor if argnums is int
               - Tuple[Tensor, ...] if argnums is tuple
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
