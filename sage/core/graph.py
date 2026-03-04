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


class TorchChoice(nn.Module):
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


class _FusedSequential(nn.Module):
    def __init__(self, modules: List[nn.Module]):
        super().__init__()

        # Flatten nested _FusedSequential modules
        flat_modules = []
        for m in modules:
            if isinstance(m, _FusedSequential):
                flat_modules.extend(m.modules_list)
            else:
                flat_modules.append(m)

        self.modules_list = nn.ModuleList(flat_modules)

    def forward(self, x=None, *args, **kwargs):
        # Fully vectorized through sequential calls
        for module in self.modules_list:
            if x is None:
                x = module(*args, **kwargs)
            else:
                x = module(x, *args, **kwargs)
        return x


class _FusedChoice(nn.Module):
    def __init__(self, modules: List[nn.Module], probs: torch.Tensor):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.register_buffer("probs", probs)

    def forward(self, x, generator=None):
        B = x.shape[0]
        device = x.device
        probs = self.probs.to(device)
        dist = torch.distributions.Categorical(probs)
        choices = dist.sample((B,), generator=generator)  # (B,)

        # Allocate output
        output = torch.empty_like(x)

        # Compute all branch outputs
        branch_outputs = []
        for module in self.modules_list:
            branch_outputs.append(module(x))  # shape: (B, ...)

        # Stack along new dimension (B, num_branches, features)
        stacked = torch.stack(branch_outputs, dim=1)

        # Use choices to select the correct branch per sample
        choices_expand = choices.view(B, 1, *([1] * (x.ndim - 1))).expand_as(stacked)
        output = torch.gather(stacked, 1, choices_expand).squeeze(1)

        return output


class _FusedGenerator(nn.Module):
    def __init__(self, modules: List[Union[nn.Module, Callable]], combiner=None):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.combiner = combiner

    def forward(self, *args, **kwargs):
        outputs = []
        for m in self.modules_list:
            if isinstance(m, nn.Module):
                out = m(*args, **kwargs)
            else:
                out = m(*args, **kwargs)
            outputs.append(out)

        if self.combiner is not None:
            return self.combiner(*outputs)
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)


def _compile_module(module: nn.Module) -> nn.Module:
    """
    Recursively converts module to a compiled version.
    """

    # TorchSequential -> fuse contained modules
    if isinstance(module, TorchSequential):
        compiled = [_compile_module(m) for m in module.modules_list]
        return _FusedSequential(compiled)

    # Generator -> fuse each branch and combiner
    elif isinstance(module, Generator):
        compiled = [
            _compile_module(m) if isinstance(m, nn.Module) else m
            for m in module.modules_list
        ]
        return _FusedGenerator(compiled, module.combiner)

    # TorchChoice -> fuse each branch
    elif isinstance(module, TorchChoice):
        compiled_branches = [_compile_module(m) for m in module.modules_list]
        return _FusedChoice(compiled_branches, module.probs)

    # Base nn.Module -> return as-is
    else:
        return module


class SageGraph(nn.Module):
    """
    High-performance DAG / pipeline wrapper.

    This class supports two independent optimisation stages:

    1. Structural fusion (fuse=True)
       - Recursively flattens nested TorchSequential, Generator,
         and TorchChoice modules into static fused modules.
       - Removes dynamic nesting overhead.
       - Makes the graph more compiler-friendly.

    2. Backend compilation (compile=True)
       - Applies torch.compile to the fused graph.
       - Enables kernel fusion, graph capture, and Inductor lowering.

    Parameters
    ----------
    modules : List[nn.Module]
        Root modules applied sequentially.
        The output of module[i] becomes input to module[i+1].

    fuse : bool, default=True
        If True, structurally flatten the module DAG into
        a static fused representation.

    compile : bool, default=False
        If True, apply torch.compile to the fused graph.

    compile_mode : str, default="default"
        Mode passed to torch.compile.
        Options include:
            "default"
            "reduce-overhead"
            "max-autotune"

    fullgraph : bool, default=True
        If True, require full graph capture.
        If your graph contains dynamic control flow,
        set this to False.
    """

    def __init__(
        self,
        modules: List[nn.Module],
        fuse: bool = True,
        compile: bool = False,
        compile_mode: str = "default",
        fullgraph: bool = True,
        dynamic: bool = False,
    ):
        super().__init__()

        # Store original modules for debugging / inspection
        self.original_modules = nn.ModuleList(modules)

        # Stage 1: Structural Fusion
        if fuse:
            # Recursively convert modules into fused equivalents
            fused_modules = [_compile_module(m) for m in modules]

            # Flatten into a single sequential pipeline
            root = _FusedSequential(fused_modules)
        else:
            # Use original structure (useful for debugging)
            root = TorchSequential(self.original_modules)

        # Stage 2: torch.compile (PyTorch 2.x compiler)
        if compile:
            root = torch.compile(
                root,
                mode=compile_mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )

        # Final executable graph
        self.root = root

    def forward(self, x=None, *args, **kwargs):
        """
        Forward pass through full pipeline.

        The output of each module is fed into the next.
        """
        return self.root(x, *args, **kwargs)

    # Internal helper: structural fusion
    def _fuse_modules(self, modules: List[nn.Module]) -> nn.Module:
        """
        Recursively fuse modules into a static sequential graph.
        """
        compiled_submodules = [_compile_module(m) for m in modules]
        return _FusedSequential(compiled_submodules)
