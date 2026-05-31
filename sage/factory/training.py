#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sage training loop.

SageVanillaTraining
    The standard Sage training loop.  Compiles nothing — the neural network
    model should be compiled externally with ``torch.compile(model, ...)``
    before being passed in, which is where the meaningful speedup comes from.

    Auto-multibanding: if the signal sampler has a ``selector`` attribute
    (set when ``multiband_mode='worst_case'`` is used), the noise batch is
    automatically projected to the coarse grid before injection — the user
    does not need a separate noise wrapper.

    GWBatch tracking: after combining signal + noise, the joint tensor is
    wrapped in a :class:`~sage.core.pipeline.GWBatch` that carries the
    current grid type and whitening state through the preprocessing pipeline.
    Pipeline violations (e.g. IFFT on a non-uniform grid, multirate sampling
    before TD conversion) raise :class:`~sage.core.pipeline.PipelineError`
    immediately when the offending module is called.

    The model receives ``batch.to_network_input()`` — a real float32 tensor
    whose shape is ``(B, D, T)`` for TD pipelines and ``(B, 2·D, N_coarse)``
    for FD_COARSE pipelines.
"""

# Packages
import torch

from tqdm import tqdm
from contextlib import nullcontext

# LOCAL
from sage.core.config import get_cfg
from sage.core.pipeline import GWBatch, Grid, ProcessingState
from .schedulers import ManageScheduler


class SageVanillaTraining(torch.nn.Module):
    """
    Standard Sage training loop.

    Each epoch iterates over ``num_iterations`` batches:

    1. Draw a signal batch ``(S, D, F/T)`` and a noise batch ``(B, D, F/T)``
       from the respective samplers.
    2. If the signal sampler uses worst-case multibanding, apply the selector
       to the noise batch automatically (no user action required).
    3. Scatter-inject signal waveforms at random positions in the noise batch.
    4. Wrap the combined tensor in a :class:`~sage.core.pipeline.GWBatch`
       carrying the current grid state.
    5. Pass through the preprocessing pipeline (whitening, optional IFFT and
       multirate sampling).
    6. Forward pass through the compiled model, loss, backpropagation.

    Parameters
    ----------
    signal_sampler : nn.Module
        Waveform signal sampler (e.g. :class:`IMRPhenomXAS_NRTidalv3`).
        If it exposes ``output_state`` and ``selector`` attributes (set
        automatically when ``multiband_mode='worst_case'`` is used), the
        training loop configures itself accordingly.
    noise_sampler : nn.Module
        Noise sampler (e.g. :class:`MemmapNoiseSampler`).  No changes needed
        for multibanding — the selector is applied internally.
    processor : nn.Module
        Preprocessing pipeline (e.g. :class:`~sage.core.graph.Preprocessor`
        wrapping :class:`FiducialWhitening` and optionally
        :class:`MultirateSampler`).
    model : nn.Module
        The neural network.  Should be compiled externally with
        ``torch.compile(model)`` before being passed here.
    loss_function : nn.Module
        Loss function returning a stacked component tensor.
    optimiser : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    scaler : torch.amp.GradScaler or None
        AMP gradient scaler.  Pass ``None`` to disable AMP.
    num_iterations : int
        Gradient steps per epoch.
    num_epochs : int
        Total epochs (pre-allocates the loss tracking tensor).
    scheduler_mode : str
        ``"batch"`` to step the scheduler every batch, ``"epoch"`` once per
        epoch.
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

        self.cfg = get_cfg()

        self.signal_sampler = signal_sampler
        self.noise_sampler  = noise_sampler
        self.processor      = processor
        self.model          = model.to(device=self.cfg.device, dtype=self.cfg.dtype)
        self.loss_function  = loss_function.to(
            device=self.cfg.device, dtype=self.cfg.dtype
        )
        self.optimiser  = optimiser
        self.scheduler  = ManageScheduler(scheduler, scheduler_mode)
        self.scaler     = scaler

        self.num_iterations = num_iterations
        self.num_epochs     = num_epochs

        self.num_point_estimate = len(self.cfg.do_point_estimate)
        self.num_targets        = self.num_point_estimate + 1
        self.B = self.cfg.batch_size
        self.S = int(self.cfg.batch_size * self.cfg.class_balance)

        # ── Auto-configure for multibanding ───────────────────────────────
        # Read the signal sampler's output state (FD_UNIFORM by default).
        # If the sampler uses worst_case multibanding it will expose a
        # 'selector' and an 'output_state' of Grid.FD_COARSE.
        self._initial_state = getattr(
            signal_sampler, "output_state",
            ProcessingState(Grid.FD_UNIFORM),
        )
        self._selector       = getattr(signal_sampler, "selector", None)
        self._freqs          = None
        self._coarse_indices = None

        if self._selector is not None:
            self._freqs          = self._selector.coarse_freqs
            self._coarse_indices = self._selector.coarse_indices

        # ── Loss tracking ─────────────────────────────────────────────────
        self.loss_components = torch.zeros(
            (num_epochs, self.loss_function.num_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def forward(self, nepoch):

        self.model.train()
        device = self.cfg.device

        for nbatch in tqdm(range(self.num_iterations)):

            # ── 1. Sample ─────────────────────────────────────────────────
            signal_data, signal_targets = self.signal_sampler()
            noise_data,  noise_targets  = self.noise_sampler()

            # ── 2. Auto-multiband noise (zero user action required) ───────
            # If the signal sampler uses worst-case multibanding, the signal
            # is already at N_coarse points.  Apply the same selector to the
            # full-resolution noise so both are on the same coarse grid.
            if self._selector is not None:
                noise_data = self._selector(noise_data)

            # ── 3. Pad noise targets ──────────────────────────────────────
            pad = torch.zeros(
                noise_targets.shape[0], self.num_point_estimate,
                device=device, dtype=noise_targets.dtype,
            )
            noise_targets = torch.cat((pad, noise_targets), dim=1)

            # ── 4. Random signal injection ────────────────────────────────
            idx        = torch.randperm(self.B, device=device)[: self.S]
            signal_pad = torch.zeros_like(noise_data)
            target_pad = torch.zeros(
                self.B, self.num_targets, device=device, dtype=signal_targets.dtype,
            )
            signal_pad[idx] = signal_data
            target_pad[idx] = signal_targets

            x       = noise_data + signal_pad
            targets = noise_targets + target_pad

            # ── 5. Wrap in GWBatch and preprocess ─────────────────────────
            batch = GWBatch(
                x,
                state          = self._initial_state,
                freqs          = self._freqs,
                coarse_indices = self._coarse_indices,
            )
            batch = self.processor(batch)

            # to_network_input(): TD → (B,D,T) float32 unchanged;
            #                     FD_COARSE → (B, 2D, N_coarse) float32
            net_input = batch.to_network_input()

            # ── 6. Forward + backward ─────────────────────────────────────
            self.optimiser.zero_grad(set_to_none=True)

            with (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.cfg.autocast
                else nullcontext()
            ):
                out  = self.model(net_input)
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
