#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : training.py
Description     : Short description of the file

Created on 2026-03-06 14:39:52

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
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

from .manager import CompileManager
from .schedulers import ManageScheduler


class SageVanillaTraining(torch.nn.Module):

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        loss_function,
        optimiser,
        scheduler,
        num_iterations,
        num_epochs,
        scheduler_mode="batch",
    ):
        super().__init__()

        # Get shared configs
        self.cfg = get_cfg()

        # Arranged in order of processing
        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.model = model.to(device=self.cfg.device, dtype=self.cfg.dtype)
        self.loss_function = loss_function.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )
        self.optimiser = optimiser
        self.scheduler = ManageScheduler(scheduler, scheduler_mode)

        # Training params
        self.num_iterations = num_iterations
        # NOTE: Wrapper does not care about num_epochs per se
        # But it can be used to make static tracking variables
        self.num_epochs = num_epochs

        # Gradient scaler if we use autocast
        # Since we use fp16 and not bf16, this should keep us away from dead neurons
        self.scaler = torch.amp.GradScaler("cuda") if self.cfg.autocast else None

        # Compile manager
        manager = CompileManager(
            generator=(signal_sampler, noise_sampler),
            processor=processor,
            model=model,
            loss_function=loss_function,
        )

        self.compiled_block = manager.compiled_block
        self.uncompiled_generator = manager.uncompiled_block

        # Tracking
        self.loss_components = torch.zeros(
            (num_epochs, self.loss_function.num_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def forward(self, nepoch):

        # Set model to training mode
        self.model.train()

        for nbatch in tqdm(range(self.num_iterations)):

            # Generate non-graph-safe data
            signal, noise = self.uncompiled_generator()

            # Reset the gradients of all optimised torch tensors
            self.optimiser.zero_grad(set_to_none=True)

            # Run compiled pipeline
            loss = self.compiled_block(signal, noise)

            # Backprop update using gradient scaler (if needed)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                loss.backward()

            # Clip gradients to make convergence somewhat easier
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.clip_norm,
            )

            # Optimiser update using gradient scaler (if needed)
            if self.scaler is not None:
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                self.optimiser.step()

            # Update scheduler
            self.scheduler.batch_step(nepoch, nbatch, self.num_iterations)

            # Storing total loss this epoch
            self.loss_components[nepoch] += loss.detach()

        # Average losses
        self.loss_components[nepoch] /= self.num_iterations


class SageUncompiledTraining(torch.nn.Module):

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        loss_function,
        optimiser,
        scheduler,
        num_iterations,
        num_epochs,
        scheduler_mode="batch",
    ):
        super().__init__()

        # Shared config
        self.cfg = get_cfg()

        # Components
        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.model = model.to(device=self.cfg.device, dtype=self.cfg.dtype)
        self.loss_function = loss_function.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )

        self.optimiser = optimiser
        self.scheduler = ManageScheduler(scheduler, scheduler_mode)

        # Training params
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs

        # Gradient scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.cfg.autocast else None

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

        self.model.train()

        for nbatch in tqdm(range(self.num_iterations)):

            # Generate batches
            signal_data, signal_targets = self.signal_sampler()
            noise_data, noise_targets = self.noise_sampler()

            device = self.cfg.device

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

            # Optimiser reset
            self.optimiser.zero_grad(set_to_none=True)

            # Forward pass
            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.autocast
                else nullcontext()
            ):
                out = self.model(x)
                loss = self.loss_function(out, targets)

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.cfg.clip_norm,
            )

            # Optimiser step
            if self.scaler is not None:
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                self.optimiser.step()

            # Scheduler
            self.scheduler.batch_step(nepoch, nbatch, self.num_iterations)

            # Loss tracking
            self.loss_components[nepoch] += loss.detach()

        # Average loss
        self.loss_components[nepoch] /= self.num_iterations
