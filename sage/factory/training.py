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

from contextlib import nullcontext

# LOCAL
from sage.core.config import get_cfg


class SageVanillaTraining:

    def __init__(
        self,
        data_generator,
        model,
        loss_function,
        optimiser,
        scheduler,
        num_iterations,
        num_epochs,
    ):
        # Get shared configs
        self.cfg = get_cfg()

        # Arranged in order of processing
        self.data_generator = data_generator
        self.model = model
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.scheduler = scheduler

        # Training params
        self.num_iterations = num_iterations
        # NOTE: Wrapper does not care about num_epochs per se
        # But it can be used to make static tracking variables
        self.num_epochs = num_epochs

        # Tracking
        self.loss_components = torch.zeros(
            (num_epochs, self.loss_function.num_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def forward(self, nepoch):

        # Set model to training mode
        self.model.train()

        for nbatch in range(self.num_iterations):

            # Sample from data generator
            x, targets = self.data_generator()

            # Reset the gradients of all optimised torch tensors
            self.optimiser.zero_grad(set_to_none=True)

            with (
                torch.autocast(device_type=self.cfg.device.type)
                if self.cfg.autocast
                else nullcontext()
            ):
                out = self.model(x)
                loss = self.loss_function(out, targets)
                self.loss_function.backward()

                # Clip gradients to make convergence somewhat easier
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.clip_norm
                )

            self.optimiser.step()
            self.scheduler.batch_step(nepoch, nbatch, self.num_iterations)

            # Storing total loss this epoch
            self.loss_components[nepoch] += loss.detach()

        # Average losses
        self.loss_components[nepoch] /= self.num_iterations
