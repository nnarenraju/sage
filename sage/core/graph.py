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


class Preprocessor(nn.Module):

    def __init__(self, modules):
        super().__init__()
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


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


class NoisySignalGenerator(nn.Module):

    def __init__(self, signal_sampler: nn.Module, noise_sampler: nn.Module):
        super().__init__()
        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler

    def sample_signal(self):
        return self.signal_sampler()

    def sample_noise(self):
        return self.noise_sampler()

    @property
    def signal_ready(self):
        return getattr(self.signal_sampler, "GRAPH_READY", True)

    @property
    def noise_ready(self):
        return getattr(self.noise_sampler, "GRAPH_READY", True)


class AddSources(nn.Module):
    """
    Merges signal and noise sources.

    Possible configurations:
        - Both internal
        - Only signal internal (noise injected)
        - Only noise internal (signal injected)
        - Neither internal (both injected)

    Forward contract:
        forward(external=None)
    """

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
    ):
        super().__init__()

        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler

        self.has_signal = signal_sampler is not None
        self.has_noise = noise_sampler is not None

    def forward(self, external=None):

        # Internal sampling
        if self.has_signal:
            signal = self.signal_sampler()
        else:
            signal = external

        if self.has_noise:
            noise = self.noise_sampler()
        else:
            noise = external

        return signal + noise


## Base: SageGraph pipeline builder ##


class SageGraph(nn.Module):

    def __init__(
        self,
        modules: List[nn.Module],
        compile: bool = False,
        compile_mode: str = "default",
        fullgraph: bool = True,
        dynamic: bool = False,
    ):
        super().__init__()

        assert len(modules) == 2
        generator, preprocess = modules

        assert isinstance(generator, NoisySignalGenerator)
        assert isinstance(preprocess, TorchSequential)

        self.generator = generator
        self.preprocess = preprocess

        self.signal_ready = generator.signal_ready
        self.noise_ready = generator.noise_ready
        self.preprocess_ready = all(
            getattr(m, "GRAPH_READY", True) for m in preprocess.modules_list
        )

        # If preprocess not ready -> disable compile
        self.do_compile = compile and self.preprocess_ready

        if not self.do_compile:
            self.compiled_block = None
            return

        # ===== CASE ANALYSIS =====

        if self.signal_ready and self.noise_ready:
            # Compile full graph
            add_node = AddSources(
                generator.signal_sampler,
                generator.noise_sampler,
            )
            block = nn.Sequential(add_node, preprocess)

        elif self.signal_ready and not self.noise_ready:
            # Compile signal + preprocess
            add_node = AddSources(
                generator.signal_sampler,
                None,
            )
            block = nn.Sequential(add_node, preprocess)

        elif not self.signal_ready and self.noise_ready:
            # Compile noise + preprocess
            add_node = AddSources(
                None,
                generator.noise_sampler,
            )
            block = nn.Sequential(add_node, preprocess)

        else:
            # Compile preprocess only
            add_node = AddSources(None, None)
            block = nn.Sequential(add_node, preprocess)

        self.compiled_block = torch.compile(
            block,
            mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

    def forward(self):

        if not self.do_compile:
            signal = self.generator.sample_signal()
            noise = self.generator.sample_noise()
            x = signal + noise
            return self.preprocess(x)

        if self.signal_ready and not self.noise_ready:
            noise = self.generator.sample_noise()
            return self.compiled_block(noise)

        if not self.signal_ready and self.noise_ready:
            signal = self.generator.sample_signal()
            return self.compiled_block(signal)

        if not self.signal_ready and not self.noise_ready:
            signal = self.generator.sample_signal()
            noise = self.generator.sample_noise()
            return self.compiled_block(signal + noise)

        return self.compiled_block()
