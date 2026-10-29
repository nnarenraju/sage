#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sage validation loop.

SageVanillaValidation
    Mirrors SageVanillaTraining exactly except the model runs in eval +
    inference_mode, and per-epoch outputs (network predictions, targets,
    raw parameters) are saved to an HDF5 file for offline diagnostics
    (ROC curves, parameter recovery, etc.).
"""

import sys
import time
import torch

from tqdm import tqdm
from contextlib import nullcontext

from sage.architecture.custom_losses import width_to_sigma
from sage.core.logger import get_logger, format_duration as _fmt_duration
from sage.core.pipeline import GWBatch
from sage.factory.training import assemble_batch
from sage.utils.atomic_io import append_h5

logger = get_logger(__name__)


def save_validation(name, output, target, params, param_names, signal_idx, savepath):
    """
    Append one validation pass's results to an HDF5 file.

    Creates a group ``name`` inside ``savepath`` containing four
    gzip-compressed datasets.

    Parameters
    ----------
    name : str
        Group name: ``epoch_NNNN`` for a regular pass, ``final`` for the
        check of ``best.pt`` after training.
    output : torch.Tensor, shape ``(N, 1 + 2*num_pe)``
        Network outputs: ranking statistic, predicted means, predicted sigmas
        (in physical units).
    target : torch.Tensor, shape ``(N, num_pe + 1)``
        Ground-truth targets (regression values + class label).
    params : torch.Tensor, shape ``(N, num_params)``
        Raw waveform parameters (``theta``) returned by the signal sampler.
    param_names : list of str
        Name of each ``params`` column, recorded as its ``names`` attribute.
    signal_idx : torch.Tensor, shape ``(num_iter, S)``
        Batch positions where signals were injected for each iteration.
    savepath : str
        Path to the HDF5 output file.
    """
    # Appended in place: this file reaches ~300 MB, so snapshotting it every
    # epoch (atomic_h5) would move tens of GB per run to protect a diagnostic
    # that training does not read back. append_h5 instead absorbs a kill on the
    # NEXT open, quarantining a damaged file so it cannot crash the run.
    with append_h5(savepath) as f:
        if name in f:
            # Overwrite a re-validated / restarted epoch rather than crashing on
            # a pre-existing validation_data.h5 (e.g. a fresh run reusing an
            # export dir, or re-running the same epoch).
            del f[name]
        grp = f.create_group(name)
        grp.create_dataset("network_output", data=output.numpy(), compression="gzip")
        grp.create_dataset("network_target", data=target.numpy(), compression="gzip")
        grp.create_dataset("signal_params",  data=params.numpy(),  compression="gzip")
        grp["signal_params"].attrs["names"] = param_names
        grp.create_dataset("signal_idx",     data=signal_idx.numpy(), compression="gzip")


class SageVanillaValidation(torch.nn.Module):
    """
    Standard Sage validation loop with full diagnostic output saving.

    Mirrors :class:`~sage.factory.training.SageVanillaTraining` but:

    * Runs the model in ``eval`` + ``inference_mode``.
    * Calls ``signal_sampler(return_theta=True)`` to obtain raw waveform
      parameters for parameter-recovery diagnostics.
    * Unscales predicted means back to physical units.
    * Converts the raw widths to sigma with the loss's mapping and bounds
      (:func:`~sage.architecture.custom_losses.width_to_sigma`), then
      unscales to physical units.
    * Writes each pass's results to ``results_path`` (``results/validation.h5``).

    GWBatch tracking works as in the training loop.

    Resume note
    -----------
    The *trained model* is fully resume-deterministic, but the validation
    *loss curve* is not bit-reproducible across a resume. The val signal/noise
    samplers and the injection permutation's generator are seeded once at
    construction and advance per call, so a run resuming at epoch K draws
    different validation realisations than a fresh run did at epoch K. This is only Monte-Carlo
    sampling noise on an unbiased estimate of the same fixed-distribution loss;
    it never affects the trained weights, only the exact val-loss value (and
    hence a marginal, MC-level wobble in best.pt selection). Making the curve
    bit-identical across resume would require per-epoch reseeding of the
    (threaded) val samplers; deliberately not done.

    Parameters
    ----------
    signal_sampler, noise_sampler, processor, model, loss_function
        Validation samplers, and the objects passed to the training class.
    num_iterations : int
        Batches per validation pass.
    num_epochs : int
        Total epochs (pre-allocates the ``loss_components`` tracking tensor).
    batch_size : int
        Windows per batch (the training one).
    signal_batch_size : int
        Windows per batch that receive a signal (the training one).
    num_estimates : int
        Estimated parameters (``signal.targets``).
    placement : torch.Generator
        Generator of the signal positions in a batch, on ``device``.
    device : str or torch.device
        Device of the batches.
    dtype : torch.dtype
        Dtype of the loss tracking tensor.
    autocast_dtype : torch.dtype or None
        Autocast dtype on ``device``; None runs without autocast.
    results_path : Path
        HDF5 file the passes are appended to.
    """

    def __init__(
        self,
        signal_sampler,
        noise_sampler,
        processor,
        model,
        loss_function,
        num_iterations: int,
        num_epochs: int,
        batch_size: int,
        signal_batch_size: int,
        num_estimates: int,
        placement: torch.Generator,
        device,
        dtype: torch.dtype,
        autocast_dtype,
        results_path,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.results_path = results_path

        self.signal_sampler = signal_sampler
        self.noise_sampler  = noise_sampler
        self.processor      = processor
        self.model          = model
        self.loss_function  = loss_function

        self.num_iterations = num_iterations
        self.num_epochs     = num_epochs
        self.batch_size        = batch_size
        self.signal_batch_size = signal_batch_size
        self.num_estimates     = num_estimates
        self.placement         = placement

        # ── Loss tracking ─────────────────────────────────────────────────
        self.loss_components = torch.zeros(
            (num_epochs, self.loss_function.num_components), device=self.device, dtype=dtype,
        )

    def forward(self, nepoch, name=None):
        """
        Run one validation pass and append its results.

        Parameters
        ----------
        nepoch : int
            Row of ``loss_components`` the pass fills.
        name : str or None
            Results group; ``epoch_NNNN`` when None.
        """
        self.model.eval()

        diagnostics = {
            "signal_params":  [],
            "signal_idx":     [],
            "network_output": [],
            "network_target": [],
        }

        # See the note in training.forward: a tqdm bar on a non-TTY fills the
        # job's .out file with carriage-return redraws.
        interactive = sys.stdout.isatty()
        t_epoch = time.time()
        logger.info(
            "Epoch %d | validating: %d iterations", nepoch, self.num_iterations
        )

        with torch.inference_mode():

            for _ in tqdm(
                range(self.num_iterations),
                disable=not interactive,
                desc=f"epoch {nepoch} val",
                leave=False,
            ):

                # ── 1. Sample ─────────────────────────────────────────────
                signal_data, signal_targets, theta = self.signal_sampler(
                    return_theta=True
                )
                noise_data, noise_targets = self.noise_sampler()

                # ── 2. Random signal injection ────────────────────────────
                x, targets, idx = assemble_batch(
                    signal_data, signal_targets, noise_data, noise_targets,
                    self.signal_batch_size, self.num_estimates, self.placement,
                )
                diagnostics["signal_idx"].append(idx.cpu())

                # ── 3. Wrap in GWBatch and preprocess ─────────────────────
                batch     = self.processor(GWBatch(x, state=self.processor.input_state))
                net_input = batch.to_network_input()

                # ── 4. Forward pass ───────────────────────────────────────
                with (
                    torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
                    if self.autocast_dtype is not None
                    else nullcontext()
                ):
                    out  = self.model(net_input)
                    loss = self.loss_function(out, targets)

                # ── 5. Diagnostics ────────────────────────────────────────
                network_output = torch.cat([*out], dim=1)

                ranking   = network_output[:, 0:1]
                mu_scaled = network_output[:, 1 : 1 + self.num_estimates]
                mu_phys   = self.signal_sampler.param_sampler.unscale_targets(mu_scaled)
                # Heteroscedastic models also emit a raw sigma per point estimate
                # (columns after the means); mean-only models (e.g. with BCE_PE_Huber)
                # do not -- their network_output has no sigma columns, so export
                # ranking + mu only (no sigma_phys) instead of crashing on an empty slice.
                if network_output.shape[1] >= 1 + 2 * self.num_estimates:
                    sigma_raw = network_output[
                        :, 1 + self.num_estimates : 1 + 2 * self.num_estimates
                    ]
                    # The loss's own mapping and bounds, so exported sigma matches
                    # the trained parameterisation.
                    sigma_scaled = width_to_sigma(
                        sigma_raw, self.loss_function.sigma_min, self.loss_function.sigma_max
                    )
                    sigma_phys = sigma_scaled * self.signal_sampler.param_sampler._target_scales
                    network_output = torch.cat([ranking, mu_phys, sigma_phys], dim=1)
                else:
                    network_output = torch.cat([ranking, mu_phys], dim=1)

                diagnostics["network_output"].append(network_output.cpu())
                diagnostics["network_target"].append(targets.cpu())
                diagnostics["signal_params"].append(theta.cpu())

                self.loss_components[nepoch] += loss.detach()

        self.loss_components[nepoch] /= self.num_iterations

        comps = self.loss_components[nepoch].detach().cpu()
        logger.info(
            "Epoch %d | val   loss %.5f | components [%s] | %s",
            nepoch,
            comps[0].item(),
            ", ".join(f"{c:.5f}" for c in comps[1:].tolist()),
            _fmt_duration(time.time() - t_epoch),
        )

        # ── Save diagnostic outputs ───────────────────────────────────────
        network_output = torch.vstack(diagnostics["network_output"])
        network_target = torch.vstack(diagnostics["network_target"])
        signal_params  = torch.vstack(diagnostics["signal_params"])
        signal_idx     = torch.stack(diagnostics["signal_idx"])

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        save_validation(
            name or f"epoch_{nepoch:04d}", network_output, network_target,
            signal_params, self.signal_sampler.param_sampler.param_names, signal_idx, self.results_path,
        )
