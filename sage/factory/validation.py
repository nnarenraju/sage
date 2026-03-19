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
import os
import h5py
import torch

from tqdm import tqdm
from contextlib import nullcontext

# LOCAL
from sage.core.config import get_cfg

from .manager import CompileManager


def save_validation(nepoch, output, target, params, savepath):

    with h5py.File(savepath, "a") as f:

        grp = f.create_group(f"epoch_{nepoch:04d}")

        grp.create_dataset("network_output", data=output.numpy(), compression="gzip")
        grp.create_dataset("network_target", data=target.numpy(), compression="gzip")
        grp.create_dataset("signal_params", data=params.numpy(), compression="gzip")


class SageVanillaValidation(torch.nn.Module):

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

        self.cfg = get_cfg()

        # Components
        self.signal_sampler = signal_sampler
        self.noise_sampler = noise_sampler
        self.processor = processor
        self.model = model.to(device=self.cfg.device, dtype=self.cfg.dtype)
        self.loss_function = loss_function.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )

        # Params
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs

        # Compile manager (reuse pattern)
        manager = CompileManager(
            generator=(signal_sampler, noise_sampler),
            processor=processor,
            model=model,
            loss_function=loss_function,
            training=False,
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

        self.model.eval()

        with torch.inference_mode():

            for _ in tqdm(range(self.num_iterations)):

                # Generate non-graph-safe data
                signal, noise = self.uncompiled_generator()

                # Forward (compiled)
                loss = self.compiled_block(signal, noise)

                # Track
                self.loss_components[nepoch] += loss.detach()

        self.loss_components[nepoch] /= self.num_iterations


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

        # Diagnostics
        save = {}
        save["signal_params"] = []
        save["network_output"] = []
        save["network_target"] = []

        with torch.inference_mode():

            for _ in tqdm(range(self.num_iterations)):

                # Generate batches
                signal_data, signal_targets, theta = self.signal_sampler(
                    return_theta=True
                )
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

                # Save results
                network_output = torch.cat([*out], dim=1)
                save["network_output"].append(network_output.cpu())
                save["network_target"].append(targets.cpu())
                save["signal_params"].append(theta.cpu())

                # Track losses
                self.loss_components[nepoch] += loss.detach()

        # Average loss
        self.loss_components[nepoch] /= self.num_iterations

        # Stack and save
        network_output = torch.stack(save["network_output"])
        network_target = torch.stack(save["network_target"])
        signal_params = torch.stack(save["signal_params"])

        savepath = os.path.join(self.cfg.export_dir, "validation_data.h5")
        save_validation(nepoch, network_output, network_target, signal_params, savepath)
