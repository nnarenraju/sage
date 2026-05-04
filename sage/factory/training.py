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
import torch.nn.functional as F

from tqdm import tqdm
from contextlib import nullcontext

# LOCAL
from sage.core.config import get_cfg

from .manager import CompileManager
from .schedulers import ManageScheduler


class SageVanillaTraining(torch.nn.Module):
    """
    Compiled training loop using the :class:`~sage.factory.manager.CompileManager`
    pattern.

    The signal and noise generators are compiled together with the
    preprocessing and forward pass where possible (controlled by each
    sampler's ``GRAPH_READY`` flag).  The optimizer step and loss
    backpropagation always run in eager mode.

    Parameters
    ----------
    signal_sampler : nn.Module
        Waveform signal sampler (e.g. :class:`IMRPhenomPv2`).
    noise_sampler : nn.Module
        Noise sampler (e.g. :class:`MemmapNoiseSampler`).
    processor : nn.Module
        Preprocessing pipeline (e.g. :class:`Preprocessor` wrapping
        :class:`FiducialWhitening` + :class:`MultirateSampler`).
    model : nn.Module
        The neural network to train.
    loss_function : nn.Module
        Loss function that returns a stacked component tensor.
    optimiser : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    num_iterations : int
        Gradient steps per epoch (typically ``n_samples // batch_size``).
    num_epochs : int
        Total epochs (pre-allocates the loss tracking tensor).
    scheduler_mode : str
        ``"batch"`` to step the scheduler every batch, ``"epoch"`` to step
        once per epoch.
    """

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
                self.scaler.scale(loss[0]).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                loss[0].backward()

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
    """
    Uncompiled training loop with explicit signal-injection batch construction.

    Builds each training batch manually:

    1. Draw a signal batch (S samples) and a noise batch (B samples).
    2. Inject the S signals into random positions of the noise batch.
    3. Preprocess the combined batch.
    4. Run the forward pass and compute the loss.
    5. Backpropagate, clip gradients, and step the optimiser.

    This is the reference implementation used before ``torch.compile`` was
    introduced, and serves as the base class pattern for
    :class:`SageHardMiningTraining`.

    Parameters
    ----------
    signal_sampler : nn.Module
        Waveform signal sampler (e.g. :class:`IMRPhenomPv2`).
    noise_sampler : nn.Module
        Noise sampler (e.g. :class:`MemmapNoiseSampler`).
    processor : nn.Module
        Preprocessing pipeline.
    model : nn.Module
        The neural network to train.
    loss_function : nn.Module
        Loss function that returns a stacked component tensor.
    optimiser : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    scaler : torch.amp.GradScaler
        AMP gradient scaler.  Pass ``None`` to disable AMP.
    num_iterations : int
        Gradient steps per epoch.
    num_epochs : int
        Total epochs (pre-allocates the loss tracking tensor).
    scheduler_mode : str
        ``"batch"`` or ``"epoch"``.
    """

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        loss_function,
        optimiser,
        scheduler,
        scaler,
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
        self.scaler = scaler

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
                self.scaler.scale(loss[0]).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                loss[0].backward()

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


# ---------------------------------------------------------------------------
# Hard-mining training loop
# ---------------------------------------------------------------------------

class SageHardMiningTraining(torch.nn.Module):
    """
    Training loop that augments each batch with hard background examples.

    Each batch replaces a controlled fraction of background slots with
    windows drawn from a pre-mined buffer of the highest-ranking noise
    samples (false-alarm candidates the model currently struggles to
    reject).  The buffer is repopulated every `mine_every_n_epochs` epochs
    (and once before epoch 0).

    Parameters
    ----------
    hard_noise_frac : float
        Fraction of background slots in each batch replaced by buffered
        hard noise (pre-processed tensors).
    mine_every_n_epochs : int
        Mining pass frequency. 0 = mine only before epoch 0.
    """

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        loss_function,
        optimiser,
        scheduler,
        scaler,
        miner,
        hard_noise_buffer,
        num_iterations:      int,
        num_epochs:          int,
        scheduler_mode:      str   = "batch",
        hard_noise_frac:     float = 0.15,
        mine_every_n_epochs: int   = 5,
    ):
        super().__init__()

        self.cfg = get_cfg()

        self.signal_sampler    = signal_sampler
        self.noise_sampler     = noise_sampler
        self.processor         = processor
        self.model             = model.to(device=self.cfg.device, dtype=self.cfg.dtype)
        self.loss_function     = loss_function.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )
        self.optimiser         = optimiser
        self.scheduler         = ManageScheduler(scheduler, scheduler_mode)
        self.scaler            = scaler

        self.miner             = miner
        self.hard_noise_buffer = hard_noise_buffer

        self.num_iterations      = num_iterations
        self.num_epochs          = num_epochs
        self.hard_noise_frac     = hard_noise_frac
        self.mine_every_n_epochs = mine_every_n_epochs

        self.num_point_estimate = len(self.cfg.do_point_estimate)
        self.num_targets        = self.num_point_estimate + 1
        self.B = self.cfg.batch_size
        self.S = int(self.cfg.batch_size * self.cfg.class_balance)

        self.loss_components = torch.zeros(
            (num_epochs, self.loss_function.num_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def forward(self, nepoch: int):

        # ── Mining pass ───────────────────────────────────────────────
        do_mine = (nepoch == 0) or (
            self.mine_every_n_epochs > 0
            and nepoch % self.mine_every_n_epochs == 0
        )
        if do_mine:
            print(f"\nEpoch {nepoch}: running hard-noise mining pass …")
            self.miner.mine(
                model         = self.model,
                noise_sampler = self.noise_sampler,
                processor     = self.processor,
                device        = self.cfg.device,
                autocast      = self.cfg.autocast,
            )

        self.model.train()
        device = self.cfg.device
        dtype  = self.cfg.dtype

        for nbatch in tqdm(range(self.num_iterations)):

            # ── 1. Sample ─────────────────────────────────────────────
            signal_data, signal_targets = self.signal_sampler()
            noise_data,  noise_targets  = self.noise_sampler()

            # Pad noise targets to (B, num_pe + 1)
            pad = torch.zeros(
                noise_targets.shape[0], self.num_point_estimate,
                device=device, dtype=noise_targets.dtype,
            )
            noise_targets = torch.cat((pad, noise_targets), dim=1)

            # ── 2. Build batch (signal injection) ─────────────────────
            idx        = torch.randperm(self.B, device=device)[: self.S]
            signal_pad = torch.zeros_like(noise_data)
            target_pad = torch.zeros(
                self.B, self.num_targets, device=device, dtype=signal_targets.dtype,
            )
            signal_pad[idx] = signal_data
            target_pad[idx] = signal_targets

            x       = noise_data + signal_pad
            targets = noise_targets + target_pad

            # ── 3. Preprocess ─────────────────────────────────────────
            x = self.processor(x)

            # ── 4. Hard noise injection (post-preprocessing) ──────────
            if self.hard_noise_buffer.is_ready:
                bg_pos = torch.where(targets[:, -1] < 0.5)[0]
                n_hard = min(
                    int(self.B * self.hard_noise_frac),
                    len(bg_pos),
                    len(self.hard_noise_buffer),
                )
                if n_hard > 0:
                    replace_pos    = bg_pos[torch.randperm(len(bg_pos))[:n_hard]]
                    hard_x         = self.hard_noise_buffer.sample(n_hard, device, dtype)
                    x[replace_pos] = hard_x

            # ── 5. Forward + backward ─────────────────────────────────
            self.optimiser.zero_grad(set_to_none=True)

            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.autocast
                else nullcontext()
            ):
                out  = self.model(x)
                loss = self.loss_function(out, targets)

            if self.scaler is not None:
                self.scaler.scale(loss[0]).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                loss[0].backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.clip_norm,
            )

            if self.scaler is not None:
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                self.optimiser.step()

            self.scheduler.batch_step(nepoch, nbatch, self.num_iterations)
            self.loss_components[nepoch] += loss.detach()

        self.loss_components[nepoch] /= self.num_iterations
