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
    """
    Sequential preprocessing pipeline for gravitational-wave data.

    Chains an ordered list of ``nn.Module`` transforms into a single
    :class:`torch.nn.Sequential` block.  Each module receives the output
    of the previous one.  The typical Sage pipeline is::

        Preprocessor([FiducialWhitening(), MultirateSampler(...)])

    This is the object passed as ``processor`` to all training and mining
    classes.  Its :meth:`forward` is called on the concatenated
    signal-plus-noise frequency-domain batch produced by the noise sampler.

    Parameters
    ----------
    modules : list of nn.Module
        Ordered preprocessing steps.  All must accept and return a
        ``torch.Tensor`` so they can be composed with ``nn.Sequential``.

    Input / Output
    --------------
    Accepts whatever tensor shape the first module in ``modules`` expects
    (typically ``(B, D, F)`` complex64 for FD strain data) and returns
    whatever the last module produces (typically ``(B, D, L_compressed)``
    float32 after whitening and multirate decimation).
    """

    def __init__(self, modules):
        super().__init__()
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (shape depends on the first module; usually
            ``(B, D, F)`` complex64 FD strain).

        Returns
        -------
        torch.Tensor
            Preprocessed tensor (shape depends on the last module; usually
            ``(B, D, L_compressed)`` float32 whitened, multi-rate sampled).
        """
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
    """
    Paired signal + noise data source for training.

    Wraps a signal sampler and a noise sampler into a single module.
    Both samplers are queried independently; the caller is responsible for
    combining their outputs (typically ``signal + noise``).

    The ``GRAPH_READY`` attribute on each sampler is read to decide whether
    that sampler can be included inside a ``torch.compile`` graph.  Samplers
    backed by async prefetch queues (e.g. :class:`MemmapNoiseSampler`) set
    ``GRAPH_READY = False`` because their Python-side queue pop cannot be
    traced by the compiler.

    Parameters
    ----------
    signal_sampler : nn.Module
        Callable that returns ``(signal_fd, signal_targets)``.
    noise_sampler : nn.Module
        Callable that returns ``(noise_fd, noise_targets)``.
    """

    def __init__(self, signal_sampler: nn.Module, noise_sampler: nn.Module):
        super().__init__()
        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler

    def sample_signal(self):
        """Draw one signal batch from the signal sampler."""
        return self.signal_sampler()

    def sample_noise(self):
        """Draw one noise batch from the noise sampler."""
        return self.noise_sampler()

    @property
    def signal_ready(self):
        """True if the signal sampler is safe to include in a compiled graph."""
        return getattr(self.signal_sampler, "GRAPH_READY", True)

    @property
    def noise_ready(self):
        """True if the noise sampler is safe to include in a compiled graph."""
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
    """
    Adaptive pipeline builder that compiles as much of the data graph as
    possible given each sampler's ``GRAPH_READY`` capability flag.

    Four compilation cases are handled automatically:

    * Both signal and noise graph-ready → compile signal + noise + preprocess.
    * Signal only graph-ready → compile signal + preprocess; noise injected.
    * Noise only graph-ready → compile noise + preprocess; signal injected.
    * Neither graph-ready → compile preprocess only; both injected externally.

    If ``compile=False`` or the preprocessing pipeline is not graph-ready,
    all modules run eagerly without compilation.

    Parameters
    ----------
    modules : list of nn.Module, length 2
        ``[NoisySignalGenerator, TorchSequential]`` — generator then preprocessor.
    compile : bool
        Whether to attempt ``torch.compile``.
    compile_mode : str
        Mode string forwarded to ``torch.compile`` (e.g. ``"default"``,
        ``"max-autotune"``).
    fullgraph : bool
        Forwarded to ``torch.compile``.  ``True`` raises an error on graph
        breaks rather than falling back to eager.
    dynamic : bool
        Forwarded to ``torch.compile``.  ``True`` enables dynamic shapes.
    """

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
