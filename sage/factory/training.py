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
    Drop-in replacement for SageUncompiledTraining that adds three mechanisms
    for improving performance at low false-alarm rates:

    1. pAUC + focal loss (via BCEWithFARLoss)
       Handled entirely inside the loss function; no changes here.

    2. Hard sample replay
       Each batch mixes a controlled fraction of pre-mined hard examples:
         • hard_noise_frac  : background windows the model currently ranks
                              highest (false-alarm candidates).
         • hard_signal_frac : signal+noise windows the model currently ranks
                              lowest (missed detections).
       Hard buffers are repopulated every `mine_every_n_epochs` epochs by
       calling miner.mine(). The first mine happens before epoch 0 so the
       buffers are populated from the start.

    3. Adversarial noise perturbation
       With probability `adv_prob`, background noise in the current batch is
       perturbed in the direction that maximises the ranking statistic (FGSM
       step, PSD-normalised). This exposes the model to noise excursions that
       look maximally signal-like under the current weights — directly
       targeting the spectral features that cause loud background triggers.

    Parameters
    ----------
    hard_noise_frac : float
        Fraction of background slots in each batch replaced by buffered hard
        noise (pre-processed tensors). 0.15 = replace 15 % of BG slots.
    hard_signal_frac : float
        Fraction of signal slots replaced by buffered hard signals.
    adv_prob : float
        Probability per batch of applying an adversarial noise perturbation to
        the current background noise before preprocessing.
    adv_eps : float
        Perturbation strength as a fraction of the local noise amplitude per
        frequency bin (PSD-normalised). Keep < 0.1 to stay physical.
    mine_every_n_epochs : int
        Mining pass frequency. Set to 0 to mine only before epoch 0.
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
        hard_signal_buffer,
        num_iterations:      int,
        num_epochs:          int,
        scheduler_mode:      str   = "batch",
        hard_noise_frac:     float = 0.15,
        hard_signal_frac:    float = 0.10,
        adv_prob:            float = 0.10,
        adv_eps:             float = 0.05,
        mine_every_n_epochs: int   = 5,
    ):
        super().__init__()

        self.cfg = get_cfg()

        self.signal_sampler     = signal_sampler
        self.noise_sampler      = noise_sampler
        self.processor          = processor
        self.model              = model.to(device=self.cfg.device, dtype=self.cfg.dtype)
        self.loss_function      = loss_function.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )
        self.optimiser          = optimiser
        self.scheduler          = ManageScheduler(scheduler, scheduler_mode)
        self.scaler             = scaler

        self.miner              = miner
        self.hard_noise_buffer  = hard_noise_buffer
        self.hard_signal_buffer = hard_signal_buffer

        self.num_iterations      = num_iterations
        self.num_epochs          = num_epochs
        self.hard_noise_frac     = hard_noise_frac
        self.hard_signal_frac    = hard_signal_frac
        self.adv_prob            = adv_prob
        self.adv_eps             = adv_eps
        self.mine_every_n_epochs = mine_every_n_epochs

        self.num_point_estimate = len(self.cfg.do_point_estimate)
        self.num_targets        = self.num_point_estimate + 1
        self.B  = self.cfg.batch_size
        self.S  = int(self.cfg.batch_size * self.cfg.class_balance)

        self.loss_components = torch.zeros(
            (num_epochs, self.loss_function.num_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    # ------------------------------------------------------------------
    # Adversarial perturbation (eager mode, no compiled-graph issues)
    # ------------------------------------------------------------------

    @torch._dynamo.disable
    def _adversarial_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a single FGSM step in the preprocessed (whitened) space.

        FiducialWhitening.forward is decorated @torch.no_grad(), so
        differentiating through the preprocessor is not possible.  Instead
        we perturb the already-whitened tensor x that the model receives,
        which is both differentiable and more directly relevant — the model
        classifies on these features, not on raw FD data.

        gradient = ∂(sum(ranking_stat)) / ∂(x)
        delta     = adv_eps * (gradient / |gradient|) * |x|

        Using torch.autograd.grad() avoids accumulating gradients into model
        parameters. The uncompiled model (_orig_mod) is used for reliability
        with torch.compile.
        """
        base_model = getattr(self.model, "_orig_mod", self.model)
        base_model.eval()

        x_leaf = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = base_model(x_leaf)

        (grad,) = torch.autograd.grad(
            outputs=out[0].float().sum(),
            inputs=x_leaf,
        )

        base_model.train()

        g_dir  = grad / grad.abs().clamp_min(1e-8)
        x_amp  = x.detach().abs().clamp_min(1e-10)
        delta  = self.adv_eps * g_dir * x_amp

        return (x_leaf + delta).detach()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def forward(self, nepoch: int):

        # ── Mining pass ───────────────────────────────────────────────
        do_mine = (nepoch == 0) or (
            self.mine_every_n_epochs > 0
            and nepoch % self.mine_every_n_epochs == 0
        )
        if do_mine:
            print(f"\nEpoch {nepoch}: running hard-sample mining pass …")
            self.miner.mine(
                model          = self.model,
                noise_sampler  = self.noise_sampler,
                signal_sampler = self.signal_sampler,
                processor      = self.processor,
                device         = self.cfg.device,
                autocast       = self.cfg.autocast,
            )

        self.model.train()
        device = self.cfg.device
        dtype  = self.cfg.dtype

        for nbatch in tqdm(range(self.num_iterations)):

            # ── 1. Sample ─────────────────────────────────────────────
            signal_data, signal_targets = self.signal_sampler()
            noise_data,  noise_targets  = self.noise_sampler()

            # Pad noise targets to (B, num_pe + 1)
            pad          = torch.zeros(
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
            targets = noise_targets + target_pad  # (B, num_pe+1)

            # ── 3. Preprocess ─────────────────────────────────────────
            x = self.processor(x)   # (B, D, L_compressed)

            # ── 4. Adversarial perturbation (post-preprocessing) ──────
            # FiducialWhitening runs under @torch.no_grad(), so we perturb
            # in the already-whitened space where the model operates.
            # Only background positions are perturbed — signal windows are
            # left intact to preserve regression targets.
            if torch.rand(1).item() < self.adv_prob:
                bg_pos = torch.where(targets[:, -1] < 0.5)[0]
                if len(bg_pos) > 0:
                    x_bg_adv = self._adversarial_noise(x[bg_pos])
                    x = x.clone()
                    x[bg_pos] = x_bg_adv

            # ── 5. Hard noise injection (post-preprocessing) ──────────
            if self.hard_noise_buffer.is_ready:
                bg_pos = torch.where(targets[:, -1] < 0.5)[0]
                n_hard = min(
                    int(self.B * self.hard_noise_frac),
                    len(bg_pos),
                    len(self.hard_noise_buffer),
                )
                if n_hard > 0:
                    replace_pos      = bg_pos[torch.randperm(len(bg_pos))[:n_hard]]
                    hard_x, _        = self.hard_noise_buffer.sample(n_hard, device, dtype)
                    x[replace_pos]   = hard_x
                    # targets remain all-zero background for these positions

            # ── 6. Hard signal injection (post-preprocessing) ─────────
            if self.hard_signal_buffer.is_ready:
                n_hard_sig = min(
                    int(self.S * self.hard_signal_frac),
                    len(self.hard_signal_buffer),
                    len(idx),
                )
                if n_hard_sig > 0:
                    sig_replace = idx[:n_hard_sig]
                    hard_sig_x, hard_sig_t = self.hard_signal_buffer.sample(
                        n_hard_sig, device, dtype
                    )
                    x[sig_replace]       = hard_sig_x
                    targets[sig_replace] = hard_sig_t

            # ── 7. Forward + backward ─────────────────────────────────
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
