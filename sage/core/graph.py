#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : modules.py
Description     : Short description of the file

Created on 2026-02-23 23:38:43

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
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
import inspect
import torch.nn as nn

from typing import List, Callable, Optional, Union


## Base: Generic sequential module ##


class TorchSequential(nn.Module):
    """
    Sequentially applies a list of modules:
        x -> module1(x) -> module2(x) -> ... -> moduleN(x)
    If x is None, modules without inputs will be called without arguments.
    """

    def __init__(self, modules: List[nn.Module]):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x=None, *args, **kwargs):
        for module in self.modules_list:
            if x is None:
                x = module(*args, **kwargs)
            else:
                x = module(x, *args, **kwargs)
        return x


## Base: Probabilistic per-sample choice ##


class TorchBatchChoice(nn.Module):
    """
    Chooses one module per sample according to provided probabilities.
    Supports batch-wise selection.
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


## Base: Generic generator ##


class Generator(nn.Module):
    """
    Wraps one or more generator modules or callables.
    Each generator produces output without input.
    Can optionally combine outputs with a combiner module (e.g., Add).
    """

    def __init__(
        self,
        modules: List[Union[nn.Module, Callable]],
        combiner: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.combiner = combiner

    def forward(self, *args, **kwargs):
        outputs = []
        for module in self.modules_list:
            if isinstance(module, nn.Module):
                sig = inspect.signature(module.forward)
                if len(sig.parameters) == 0:
                    out = module()
                else:
                    out = module(*args, **kwargs)
            else:
                sig = inspect.signature(module)
                if len(sig.parameters) == 0:
                    out = module()
                else:
                    out = module(*args, **kwargs)
            outputs.append(out)

        if self.combiner is not None:
            return self.combiner(*outputs)
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)


## Base: Add combiner ##


class Add(nn.Module):
    """
    Element-wise summation of multiple inputs.
    """

    def forward(self, *args):
        total = args[0]
        for t in args[1:]:
            total = total + t
        return total


## Base: SageGraph pipeline builder ##


class SageGraph(nn.Module):
    """
    DAG / pipeline builder. Accepts a TorchSequential as config.
    Can mix generators, sequential transforms, and probabilistic choices.
    """

    def __init__(self, root: nn.Module):
        super().__init__()
        self.root = root

    def forward(self, x=None, *args, **kwargs):
        return self.root(x, *args, **kwargs)
