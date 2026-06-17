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
        callbacks=None,
        aux_losses=None,
        balancer=None,
    ):
        super().__init__()

        self.cfg = get_cfg()

        self.signal_sampler = signal_sampler
        self.noise_sampler  = noise_sampler
        self.processor      = processor
        self.model          = model.to(device=self.cfg.device, dtype=self.cfg.dtype)

        # Multi-loss mode (aux_losses and/or a balancer given): loss_function is a
        # LossAdapter and the trainer combines main + aux, optionally
        # gradient-balanced. Otherwise plain vanilla: a raw loss called as
        # loss_function(out, targets).
        self.aux_losses = list(aux_losses) if aux_losses else []
        self.balancer   = balancer
        self._multiloss = bool(self.aux_losses) or (balancer is not None)
        if self._multiloss:
            self.loss_function = loss_function          # a LossAdapter instance
        else:
            self.loss_function = loss_function.to(
                device=self.cfg.device, dtype=self.cfg.dtype
            )
        self.optimiser  = optimiser
        self.scheduler  = ManageScheduler(scheduler, scheduler_mode)
        self.scaler     = scaler

        self.num_iterations = num_iterations
        self.num_epochs     = num_epochs

        # Training callbacks (loop hooks). Empty -> plain vanilla training.
        self.callbacks = list(callbacks) if callbacks else []

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
        # vanilla: the loss's own components; multi-loss: [total, main_total,
        # *aux_adapter_totals].
        n_components = (
            2 + len(self.aux_losses) if self._multiloss
            else self.loss_function.num_components
        )
        self.loss_components = torch.zeros(
            (num_epochs, n_components),
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )

    def _collect(self, out, targets, ctx):
        """Run the main + aux loss adapters (multi-loss mode).

        Returns ``(primary, aux_terms, adapter_totals)``: the single primary
        (reference / BCE) term, the flat list of auxiliary terms to balance, and
        each adapter's own total (for logging).
        """
        primary = None
        aux_terms = []
        totals = []
        for adapter in [self.loss_function, *self.aux_losses]:
            comps = adapter(out, targets, ctx)
            totals.append(comps[0])
            if adapter.primary_index is not None:
                primary = comps[adapter.primary_index]
            aux_terms += [comps[i] for i in adapter.aux_indices]
        return primary, aux_terms, totals

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

            # ── 4b. Per-batch callback hook (e.g. non-astro injection) ────
            # ctx is threaded to on_sample (callbacks mutate ctx['x'] /
            # ctx['targets'] and add extras like per_det_mask) and to the loss
            # adapters in multi-loss mode. Pure vanilla (no callbacks, no
            # multi-loss) -> ctx is None and x / targets are unchanged.
            ctx = None
            if self.callbacks or self._multiloss:
                ctx = {
                    "x": x, "targets": targets,
                    "signal_data": signal_data, "signal_targets": signal_targets,
                    "noise_data": noise_data, "noise_targets": noise_targets,
                }
                for cb in self.callbacks:
                    cb.on_sample(ctx, self)
                x, targets = ctx["x"], ctx["targets"]

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
                out = self.model(net_input)
                if self._multiloss:
                    primary, aux_terms, log_terms = self._collect(out, targets, ctx)
                else:
                    loss = self.loss_function(out, targets)

            # Combine the total OUTSIDE autocast (the balancer's calibration runs
            # its own forward); mirrors the consistency training loop.
            if self._multiloss:
                if self.balancer is not None:
                    warmup = (
                        (nbatch + 1) / self.num_iterations if nepoch == 0 else 1.0
                    )

                    def _recompute(o):
                        p, a, _ = self._collect(o, targets, ctx)
                        return p, a

                    total = self.balancer.combine(
                        primary, aux_terms, self.model, net_input, _recompute, warmup
                    )
                else:
                    total = primary + sum(aux_terms)
            else:
                total = loss[0]

            if self.scaler is not None:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.clip_norm,
            )

            if self.scaler is not None:
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                self.optimiser.step()

            self.scheduler.batch_step(nepoch, nbatch, self.num_iterations)
            if self._multiloss:
                # [total, main_total, *aux_adapter_totals]
                self.loss_components[nepoch] += torch.stack(
                    [total.detach()] + [t.detach() for t in log_terms]
                )
            else:
                self.loss_components[nepoch] += loss.detach()

        # ── End-of-epoch callback hook (e.g. hard-noise mining) ───────────
        for cb in self.callbacks:
            cb.on_epoch_end(nepoch, self)

        self.loss_components[nepoch] /= self.num_iterations
