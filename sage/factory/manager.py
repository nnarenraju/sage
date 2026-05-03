#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : manager.py
Description     : Short description of the file

Created on 2026-03-07 19:45:34

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
import torch.nn as nn

from typing import Iterable
from contextlib import nullcontext

# LOCAL
from sage.core.config import get_cfg


class CompileManager:
    """
    Orchestrates ``torch.compile`` for the training/validation inner loop.

    Splits the pipeline into two parts:

    * **Uncompiled block** — data generation calls (signal and noise
      samplers) that may use Python-level randomness, HDF5/memmap I/O,
      or other operations unsafe inside a ``torch.compile`` graph.  Only
      samplers whose ``GRAPH_READY`` attribute is ``False`` (the default)
      are called here.

    * **Compiled block** — everything downstream of generation (signal
      injection scatter, preprocessing, model forward, loss).  Compiled
      with ``mode="max-autotune"``, ``fullgraph=True``, ``dynamic=False``
      for maximum GPU throughput.

    The compiled block is either :class:`CompiledTrainingBlock` or
    :class:`CompiledValidationBlock` depending on the ``training`` flag.

    Parameters
    ----------
    generator : tuple[signal_sampler, noise_sampler]
        Pair of data-generation objects.  Each is tested for ``GRAPH_READY``.
    processor : Iterable
        Preprocessing pipeline (e.g. :class:`~sage.core.graph.Preprocessor`).
    model : nn.Module
        The network being trained.
    loss_function : nn.Module
        Loss module, must return a scalar or loss-component tensor.
    training : bool
        If ``True`` (default), build a :class:`CompiledTrainingBlock`;
        otherwise build a :class:`CompiledValidationBlock`.
    """

    def __init__(
        self,
        generator,
        processor: Iterable,
        model: nn.Module,
        loss_function: nn.Module,
        training: bool = True,
    ):
        # Shared config
        self.cfg = get_cfg()

        # Blocks
        self.signal_sampler, self.noise_sampler = generator
        self.processor = processor
        self.model = model
        self.loss_function = loss_function

        # Graph readiness of generators
        # Default is set to False and will come under uncompiled block
        self.signal_graph = getattr(self.signal_sampler, "GRAPH_READY", False)
        self.noise_graph = getattr(self.noise_sampler, "GRAPH_READY", False)

        # Target handling
        self.num_point_estimate = len(self.cfg.do_point_estimate)
        self.num_targets = self.num_point_estimate + 1

        self.compiled_block = self._make_compiled_block(training)

    def uncompiled_block(self):
        """
        Run the graph-unsafe data generators and return ``(signal, noise)``.

        Samplers whose ``GRAPH_READY`` flag is ``True`` are skipped (they
        will be called inside the compiled graph instead).  Returns ``None``
        for any graph-ready sampler so the compiled block can detect that
        and call it internally.

        Returns
        -------
        tuple[signal or None, noise or None]
        """
        signal = None
        noise = None

        if not self.signal_graph:
            signal = self.signal_sampler()

        if not self.noise_graph:
            noise = self.noise_sampler()

        return signal, noise

    def _make_compiled_block(self, training):

        block_cls = CompiledTrainingBlock if training else CompiledValidationBlock

        # Compile the block
        compiled_block = block_cls(
            self.signal_sampler,
            self.noise_sampler,
            self.processor,
            self.cfg.batch_size,
            int(self.cfg.batch_size * self.cfg.class_balance),
            self.num_targets,
            self.num_point_estimate,
            self.model,
            self.loss_function,
        )

        return torch.compile(
            compiled_block,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )


class CompiledTrainingBlock(nn.Module):
    """
    ``torch.compile``-compatible training inner loop.

    Handles everything inside the compiled graph:

    1. Calls graph-ready samplers (``GRAPH_READY=True``) if needed.
    2. Pads noise targets to match the signal-target width.
    3. Places signal waveforms at random positions in the noise batch using
       ``scatter_add`` (avoids Python-level indexing).
    4. Combines signal + noise, preprocesses, and runs the forward pass.
    5. Computes the loss under optional AMP autocast.

    Parameters
    ----------
    signal_sampler, noise_sampler
        Data generators.
    processor
        Preprocessing pipeline.
    B : int
        Batch size.
    S : int
        Number of signal injections per batch (= B * class_balance).
    T : int
        Total target width (num_point_estimates + 1).
    P : int
        Number of point-estimate targets.
    model : nn.Module
    loss_function : nn.Module
    """

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        B,
        S,
        T,
        P,
        model,
        loss_function,
    ):
        super().__init__()

        self.cfg = get_cfg()

        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.B, self.S, self.T, self.P = (B, S, T, P)
        self.model = model
        self.loss_function = loss_function

    def forward(self, signal, noise):

        # TODO: Conditionals inside compiled graph is bad
        # Move this outside the graph somehow
        if signal is None:
            signal = self.signal_sampler()
        if noise is None:
            noise = self.noise_sampler()

        signal_data, signal_targets = signal
        noise_data, noise_targets = noise

        device = self.cfg.device

        # Pad noise targets
        pad = torch.zeros(
            noise_targets.shape[0],
            self.P,
            device=device,
            dtype=noise_targets.dtype,
        )

        noise_targets = torch.cat((pad, noise_targets), dim=1)

        # Select indices
        # TODO: Move this outside compiled graph
        # This will likely cause graph breaks
        # DO NOT naively replace with randint; randperm ensures no replacement
        # Latter code requires no replacement to be true
        idx = torch.randperm(self.B, device=device)[: self.S]

        # Scatter signal data
        signal_pad = torch.zeros_like(noise_data)

        scatter_idx = idx.view(-1, 1, 1).expand_as(signal_data)

        # TODO: Check scatter_add for efficiency issues
        signal_pad = signal_pad.scatter_add(
            0,
            scatter_idx,
            signal_data,
        )

        # Combine signal + noise
        x = noise_data + signal_pad

        # Scatter signal targets
        target_idx = idx.view(-1, 1).expand_as(signal_targets)

        signal_target_pad = torch.zeros_like(noise_targets)

        signal_target_pad = signal_target_pad.scatter_add(
            0,
            target_idx,
            signal_targets,
        )

        # Combine targets
        targets = noise_targets + signal_target_pad

        # Preprocess
        x = self.processor(x)

        # Forward
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.cfg.autocast
            else nullcontext()
        ):
            out = self.model(x)
            loss = self.loss_function(out, targets)

        return loss


class CompiledValidationBlock(nn.Module):
    """
    ``torch.compile``-compatible validation inner loop.

    Mirrors :class:`CompiledTrainingBlock` exactly but wraps the forward
    pass in ``torch.inference_mode()`` to disable gradient tracking.  This
    is the only meaningful behavioural difference; the signal injection,
    preprocessing, and loss computation are identical.

    Parameters
    ----------
    Same as :class:`CompiledTrainingBlock`.
    """

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        B,
        S,
        T,
        P,
        model,
        loss_function,
    ):
        super().__init__()

        self.cfg = get_cfg()

        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.B, self.S, self.T, self.P = (B, S, T, P)
        self.model = model
        self.loss_function = loss_function

    def forward(self, signal, noise):

        # Keep identical behavior (including graph-unsafe parts)
        if signal is None:
            signal = self.signal_sampler()
        if noise is None:
            noise = self.noise_sampler()

        signal_data, signal_targets = signal
        noise_data, noise_targets = noise

        device = self.cfg.device

        # Pad noise targets
        pad = torch.zeros(
            noise_targets.shape[0],
            self.P,
            device=device,
            dtype=noise_targets.dtype,
        )
        noise_targets = torch.cat((pad, noise_targets), dim=1)

        # Random placement (unchanged)
        idx = torch.randperm(self.B, device=device)[: self.S]

        # Scatter signal data
        signal_pad = torch.zeros_like(noise_data)
        scatter_idx = idx.view(-1, 1, 1).expand_as(signal_data)

        signal_pad = signal_pad.scatter_add(
            0,
            scatter_idx,
            signal_data,
        )

        # Combine signal + noise
        x = noise_data + signal_pad

        # Scatter signal targets
        target_idx = idx.view(-1, 1).expand_as(signal_targets)

        signal_target_pad = torch.zeros_like(noise_targets)
        signal_target_pad = signal_target_pad.scatter_add(
            0,
            target_idx,
            signal_targets,
        )

        # Combine targets
        targets = noise_targets + signal_target_pad

        # Preprocess
        x = self.processor(x)

        # ONLY MEANINGFUL CHANGE compared to training version
        # TODO: Remove redundancy between this code and training counterpart
        # Disable autograd inside compiled graph
        with torch.inference_mode():

            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.autocast
                else nullcontext()
            ):
                out = self.model(x)
                loss = self.loss_function(out, targets)

        return loss
