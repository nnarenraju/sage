#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sage training loop.

SageVanillaTraining
    The standard Sage training loop.  Compiles nothing — the neural network
    model should be compiled externally with ``torch.compile(model, ...)``
    before being passed in, which is where the meaningful speedup comes from.

    GWBatch tracking: after combining signal + noise, the joint tensor is
    wrapped in a :class:`~sage.core.pipeline.GWBatch` that carries the
    current grid type and whitening state through the preprocessing pipeline.
    Pipeline violations (e.g. IFFT on a non-uniform grid, multirate sampling
    before TD conversion) raise :class:`~sage.core.pipeline.PipelineError`
    immediately when the offending module is called.

    The model receives ``batch.to_network_input()`` — a real float32 tensor
    of shape ``(B, D, T)``.
"""

# Packages
import sys
import time
import torch

from tqdm import tqdm
from contextlib import nullcontext

# LOCAL
from sage.core.logger import get_logger, format_duration as _fmt_duration
from sage.core.pipeline import GWBatch
from .schedulers import ManageScheduler

logger = get_logger(__name__)


def assemble_batch(signal_data, signal_targets, noise_data, noise_targets, signal_batch_size, num_estimates, generator):
    """
    Inject the signals into the noise batch at random positions.

    Parameters
    ----------
    signal_data, signal_targets : torch.Tensor
        Signals and their targets (estimates, then the class label).
    noise_data, noise_targets : torch.Tensor
        Noise batch and its class labels.
    signal_batch_size : int
        Windows that receive a signal.
    num_estimates : int
        Estimated parameters.
    generator : torch.Generator
        Generator of the positions, on the batch's device.

    Returns
    -------
    x, targets : torch.Tensor
        The batch and its targets (zero estimates in noise windows).
    idx : torch.Tensor
        Positions of the signals.
    """
    batch_size, device = noise_data.shape[0], noise_data.device
    pad = torch.zeros(noise_targets.shape[0], num_estimates, device=device, dtype=noise_targets.dtype)
    noise_targets = torch.cat((pad, noise_targets), dim=1)

    idx = torch.randperm(batch_size, device=device, generator=generator)[:signal_batch_size]
    signal_pad = torch.zeros_like(noise_data)
    target_pad = torch.zeros(batch_size, num_estimates + 1, device=device, dtype=signal_targets.dtype)
    signal_pad[idx] = signal_data
    target_pad[idx] = signal_targets
    return noise_data + signal_pad, noise_targets + target_pad, idx


class SageVanillaTraining(torch.nn.Module):
    """
    Standard Sage training loop.

    Each epoch iterates over ``num_iterations`` batches:

    1. Draw a signal batch ``(S, D, F/T)`` and a noise batch ``(B, D, F/T)``
       from the respective samplers.
    2. Scatter-inject signal waveforms at random positions in the noise batch.
    3. Wrap the combined tensor in a :class:`~sage.core.pipeline.GWBatch`
       carrying the current grid state.
    4. Pass through the preprocessing pipeline (whitening, optional IFFT and
       multirate sampling).
    5. Forward pass through the compiled model, loss, backpropagation.

    Parameters
    ----------
    signal_sampler : nn.Module
        Waveform signal sampler (e.g. :class:`IMRPhenomXAS_NRTidalv3`).
    noise_sampler : nn.Module
        Noise sampler (e.g. :class:`MemmapNoiseSampler`).
    processor : nn.Module
        Preprocessing pipeline (e.g. :class:`~sage.core.graph.Preprocessor`
        wrapping :class:`FiducialWhitening` and optionally
        :class:`MultirateSampler`).
    model : nn.Module
        The neural network.  Should be compiled externally with
        ``torch.compile(model)`` before being passed here.
    loss_function : nn.Module
        Loss function returning a stacked component tensor, total first.
    optimiser : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler
    scaler : torch.amp.GradScaler or None
        Loss scaler (fp16 only), or None.
    num_iterations : int
        Batches per epoch.
    num_epochs : int
        Total epochs (pre-allocates the loss tracking tensor).
    batch_size : int
        Windows per batch.
    signal_batch_size : int
        Windows per batch that receive a signal.
    num_estimates : int
        Estimated parameters (``signal.targets``).
    placement : torch.Generator
        Generator of the signal positions in a batch, on ``device``.
    clip_norm : float
        Gradient-norm clipping threshold.
    device : str or torch.device
        Device of the batches, model and loss.
    dtype : torch.dtype
        Dtype of the model and loss.
    autocast_dtype : torch.dtype or None
        Autocast dtype (bf16 or fp16) on ``device``; None runs without autocast.
    scheduler_mode : str
        ``"batch"`` to step the scheduler every batch, ``"epoch"`` once per
        epoch, ``"metric"`` once per epoch with the validation loss.
    callbacks : list
        Loop hooks (mining, EMA); empty for plain training.
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
        num_iterations: int,
        num_epochs: int,
        batch_size: int,
        signal_batch_size: int,
        num_estimates: int,
        placement: torch.Generator,
        clip_norm: float,
        device,
        dtype: torch.dtype,
        autocast_dtype,
        scheduler_mode: str,
        callbacks: list,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.dtype = dtype

        self.signal_sampler = signal_sampler
        self.noise_sampler  = noise_sampler
        self.processor      = processor
        self.model          = model.to(device=self.device, dtype=dtype)

        # Vanilla single loss: called as loss_function(out, targets) and
        # returning a components tensor whose [0] is the total to backprop.
        self.loss_function = loss_function.to(device=self.device, dtype=dtype)
        self.optimiser  = optimiser
        self.scheduler  = ManageScheduler(scheduler, scheduler_mode)
        self.scaler     = scaler

        self.num_iterations = num_iterations
        self.num_epochs     = num_epochs
        self.batch_size        = batch_size
        self.signal_batch_size = signal_batch_size
        self.num_estimates     = num_estimates
        self.placement         = placement
        self.clip_norm         = clip_norm
        self.autocast_dtype    = autocast_dtype

        # Training callbacks (loop hooks). Empty -> plain vanilla training.
        self.callbacks = list(callbacks)

        # ── Loss tracking ─────────────────────────────────────────────────
        n_components = self.loss_function.num_components
        self.loss_components = torch.zeros(
            (num_epochs, n_components), device=self.device, dtype=dtype,
        )

    def _default_assembly(self, signal_data, signal_targets, noise_data, noise_targets):
        """Vanilla batch assembly (:func:`assemble_batch`). Returns ``(x, targets)``.
        Skipped when an on_sample callback assembles the batch itself (no shipped
        callback does)."""
        x, targets, _ = assemble_batch(
            signal_data, signal_targets, noise_data, noise_targets, self.signal_batch_size, self.num_estimates,
            self.placement,
        )
        return x, targets

    def metric_step(self, metric):
        """
        Advance a ``"metric"``-mode scheduler after a validation pass.

        Parameters
        ----------
        metric : float
            Validation metric, e.g. the validation loss.
        """
        if self.scheduler.mode == "metric":
            self.scheduler.epoch_step(metric)

    def forward(self, nepoch):

        self.model.train()

        # A progress bar is useful interactively and actively harmful in a
        # batch job: tqdm repaints with carriage returns, which a SLURM .out
        # file records verbatim, so a long run buries its own output under
        # hundreds of MB of redraw. Show the bar on a TTY; on anything else
        # emit a periodic single line instead.
        interactive = sys.stdout.isatty()
        log_every = max(1, self.num_iterations // 10)
        t_epoch = time.time()

        logger.info(
            "Epoch %d | training: %d iterations, batch %d",
            nepoch, self.num_iterations, self.batch_size,
        )

        _iter = tqdm(
            range(self.num_iterations),
            disable=not interactive,
            desc=f"epoch {nepoch} train",
            leave=False,
        )
        for nbatch in _iter:

            # ── 1. Sample ─────────────────────────────────────────────────
            signal_data, signal_targets = self.signal_sampler()
            noise_data,  noise_targets  = self.noise_sampler()

            # ── 2. Per-batch context + on_sample callbacks ────────────────
            # ctx is built only when callbacks are present and threaded to each
            # callback's on_sample hook. A callback MAY assemble the batch by
            # setting ctx['x'] / ctx['targets'] (a generic extension point — no
            # shipped callback uses it). Pure vanilla / hard-mining -> both fall
            # through to _default_assembly below; with no callbacks ctx is None.
            ctx = None
            if self.callbacks:
                ctx = {
                    "signal_data": signal_data, "signal_targets": signal_targets,
                    "noise_data": noise_data, "noise_targets": noise_targets,
                    "x": None, "targets": None,
                }
                for cb in self.callbacks:
                    cb.on_sample(ctx, self)

            # ── 3. Default assembly (random signal injection) unless a
            #        callback already produced ctx['x'] / ctx['targets']. ────
            if ctx is not None and ctx.get("x") is not None:
                x, targets = ctx["x"], ctx["targets"]
            else:
                x, targets = self._default_assembly(signal_data, signal_targets, noise_data, noise_targets)
                if ctx is not None:
                    ctx["x"], ctx["targets"] = x, targets

            # ── 4. Wrap in GWBatch and preprocess ─────────────────────────
            batch = self.processor(GWBatch(x, state=self.processor.input_state))

            net_input = batch.to_network_input()

            # ── 5. Forward + backward ─────────────────────────────────────
            self.optimiser.zero_grad(set_to_none=True)

            with (
                torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
                if self.autocast_dtype is not None
                else nullcontext()
            ):
                out = self.model(net_input)
                loss = self.loss_function(out, targets)

            total = loss[0]

            if self.scaler is not None:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimiser)
            else:
                total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.clip_norm,
            )

            if self.scaler is not None:
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                self.optimiser.step()

            self.scheduler.batch_step()
            self.loss_components[nepoch] += loss.detach()

            # ── Progress reporting ────────────────────────────────────────
            done = nbatch + 1
            if interactive:
                if done % 50 == 0:
                    running = (self.loss_components[nepoch][0] / done).item()
                    _iter.set_postfix(loss=f"{running:.4f}")
            elif done % log_every == 0 or done == self.num_iterations:
                running = (self.loss_components[nepoch][0] / done).item()
                elapsed = time.time() - t_epoch
                rate = done / max(elapsed, 1e-9)
                logger.info(
                    "  epoch %d | %3.0f%% (%d/%d) | loss %.4f | %.1f it/s | eta %s",
                    nepoch, 100.0 * done / self.num_iterations, done,
                    self.num_iterations, running, rate,
                    _fmt_duration((self.num_iterations - done) / max(rate, 1e-9)),
                )

            # Per-batch callback hook (e.g. per-step weight EMA).
            for cb in self.callbacks:
                cb.on_batch_end(nbatch, nepoch, self)

        # ── End-of-epoch callback hook (e.g. hard-noise mining) ───────────
        for cb in self.callbacks:
            cb.on_epoch_end(nepoch, self)

        # "metric" mode steps in metric_step, once a validation metric exists.
        if self.scheduler.mode == "epoch":
            self.scheduler.epoch_step()

        self.loss_components[nepoch] /= self.num_iterations

        # One summary line per epoch. This is the record that matters when
        # reading back a multi-day run, and until now nothing was emitted at
        # all -- a 46-hour job reported no loss whatsoever.
        comps = self.loss_components[nepoch].detach().cpu()
        elapsed = time.time() - t_epoch
        try:
            lr = self.optimiser.param_groups[0]["lr"]
        except (AttributeError, IndexError, KeyError):
            lr = float("nan")
        logger.info(
            "Epoch %d | train loss %.5f | components [%s] | lr %.3e | %.1f it/s | %s",
            nepoch,
            comps[0].item(),
            ", ".join(f"{c:.5f}" for c in comps[1:].tolist()),
            lr,
            self.num_iterations / max(elapsed, 1e-9),
            _fmt_duration(elapsed),
        )
