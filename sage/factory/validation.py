#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : validation.py
Description     : Short description of the file

Created on 2026-03-06 16:43:22

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

from tqdm import tqdm
from contextlib import nullcontext

# LOCAL
from sage.core.config import get_cfg


class SageUncompiledValidation(torch.nn.Module):

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        loss_function,
        num_iterations,
        num_epochs,
    ):
        super().__init__()

        # Shared config
        self.cfg = get_cfg()

        # Components
        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.model = model
        self.loss_function = loss_function

        # Validation params
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs

        # Target structure
        self.num_point_estimate = len(self.cfg.do_point_estimate)
        self.num_targets = self.num_point_estimate + 1

        # Batch structure
        self.B = self.cfg.batch_size
        self.S = int(self.cfg.batch_size * self.cfg.class_balance)

        # Tracking
        self.loss_components = torch.zeros(
            (num_epochs, self.loss_function.num_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def forward(self, nepoch):

        device = self.cfg.device

        # Evaluation mode
        self.model.eval()

        with torch.inference_mode():

            for _ in tqdm(range(self.num_iterations)):

                # Generate batches
                signal_data, signal_targets = self.signal_sampler()
                noise_data, noise_targets = self.noise_sampler()

                # Pad noise targets
                pad = torch.zeros(
                    noise_targets.shape[0],
                    self.num_point_estimate,
                    device=device,
                    dtype=noise_targets.dtype,
                )

                noise_targets = torch.cat((pad, noise_targets), dim=1)

                # Random signal placement
                idx = torch.randperm(self.B, device=device)[: self.S]

                signal_pad = torch.zeros_like(noise_data)

                target_pad = torch.zeros(
                    self.B,
                    self.num_targets,
                    device=device,
                    dtype=signal_targets.dtype,
                )

                signal_pad[idx] = signal_data
                target_pad[idx] = signal_targets

                # Combine signal + noise
                x = noise_data + signal_pad
                targets = noise_targets + target_pad

                # Preprocess
                x = self.processor(x)

                # Forward pass
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.cfg.autocast
                    else nullcontext()
                ):
                    out = self.model(x)
                    loss = self.loss_function(out, targets)

                # Track losses
                self.loss_components[nepoch] += loss.detach()

        # Average loss
        self.loss_components[nepoch] /= self.num_iterations
