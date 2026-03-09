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

    def __init__(
        self,
        generator,
        processor: Iterable,
        model: nn.Module,
        loss_function: nn.Module,
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

        self.compiled_block = self._make_compiled_block()

    def uncompiled_block(self):
        signal = None
        noise = None

        if not self.signal_graph:
            signal = self.signal_sampler()

        if not self.noise_graph:
            noise = self.noise_sampler()

        return signal, noise

    def _make_compiled_block(self):

        # Compile the block
        compiled_block = CompiledScatterBlock(
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


class CompiledBlock(nn.Module):

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
        # Shared config
        self.cfg = get_cfg()

        # All components meant to be compiled
        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.B, self.S, self.T, self.P = (B, S, T, P)
        self.model = model
        self.loss_function = loss_function

    def forward(self, signal, noise):

        # Generate missing sampler batches
        if signal is None:
            signal = self.signal_sampler()
        if noise is None:
            noise = self.noise_sampler()

        # Unpack data and targets
        signal_data, signal_targets = signal
        noise_data, noise_targets = noise

        # Pad noise targets
        pad = torch.zeros(
            noise_targets.shape[0],
            self.P,
            device=self.cfg.device,
            dtype=noise_targets.dtype,
        )
        # shape (B, T)
        noise_targets = torch.cat((pad, noise_targets), dim=1)

        # Randomly select indices for signals
        idx = torch.randperm(self.B, device=self.cfg.device)[: self.S]

        # Prepare padded signal tensors
        signal_pad = torch.zeros_like(noise_data)
        target_pad = torch.zeros(
            self.B,
            self.T,
            device=self.cfg.device,
            dtype=signal_targets.dtype,
        )

        signal_pad[idx] = signal_data
        target_pad[idx] = signal_targets

        # Combine signal + noise
        x = noise_data + signal_pad

        # Combine targets: noise padded targets + signal targets
        targets = noise_targets
        targets[idx] = signal_targets

        # Run preprocessing
        x = self.processor(x)

        # Forward pass and loss under autocast
        with torch.autocast(device_type="cuda") if self.cfg.autocast else nullcontext():
            out = self.model(x)
            loss = self.loss_function(out, targets)

        return loss


class CompiledScatterBlock(nn.Module):

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

        # Forward
        with (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.cfg.autocast
            else nullcontext()
        ):
            out = self.model(x)
            loss = self.loss_function(out, targets)

        return loss
