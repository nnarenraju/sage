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

import os
import h5py
import torch

from tqdm import tqdm
from contextlib import nullcontext

from sage.core.config import get_cfg
from sage.core.pipeline import GWBatch, Grid, ProcessingState


def save_validation(nepoch, output, target, params, signal_idx, savepath):
    """
    Append one epoch's validation results to an HDF5 file.

    Creates a group ``epoch_{nepoch:04d}`` inside ``savepath`` containing
    four gzip-compressed datasets.

    Parameters
    ----------
    nepoch : int
        Epoch index (used as the HDF5 group name).
    output : torch.Tensor, shape ``(N, 1 + 2*num_pe)``
        Network outputs: ranking statistic, predicted means, predicted sigmas
        (in physical units).
    target : torch.Tensor, shape ``(N, num_pe + 1)``
        Ground-truth targets (regression values + class label).
    params : torch.Tensor, shape ``(N, num_params)``
        Raw waveform parameters (``theta``) returned by the signal sampler.
    signal_idx : torch.Tensor, shape ``(num_iter, S)``
        Batch positions where signals were injected for each iteration.
    savepath : str
        Path to the HDF5 output file.
    """
    with h5py.File(savepath, "a") as f:
        name = f"epoch_{nepoch:04d}"
        if name in f:
            # Overwrite a re-validated / restarted epoch rather than crashing on
            # a pre-existing validation_data.h5 (e.g. a fresh run reusing an
            # export dir, or re-running the same epoch).
            del f[name]
        grp = f.create_group(name)
        grp.create_dataset("network_output", data=output.numpy(), compression="gzip")
        grp.create_dataset("network_target", data=target.numpy(), compression="gzip")
        grp.create_dataset("signal_params",  data=params.numpy(),  compression="gzip")
        grp.create_dataset("signal_idx",     data=signal_idx.numpy(), compression="gzip")


class SageVanillaValidation(torch.nn.Module):
    """
    Standard Sage validation loop with full diagnostic output saving.

    Mirrors :class:`~sage.factory.training.SageVanillaTraining` but:

    * Runs the model in ``eval`` + ``inference_mode``.
    * Calls ``signal_sampler(return_theta=True)`` to obtain raw waveform
      parameters for parameter-recovery diagnostics.
    * Unstandardises predicted means back to physical units.
    * Converts log-variance to sigma (``σ = exp(0.5 * log_var)``).
    * Writes per-epoch results to ``{export_dir}/validation_data.h5``.

    Auto-multibanding and GWBatch tracking work identically to the training
    loop — no extra configuration needed.

    Parameters
    ----------
    signal_sampler, noise_sampler, processor, model, loss_function
        Same objects passed to the corresponding training class.
    num_iterations : int
        Number of batches per validation epoch.
    num_epochs : int
        Total epochs (pre-allocates the ``loss_components`` tracking tensor).
    """

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

        self.signal_sampler = signal_sampler
        self.noise_sampler  = noise_sampler
        self.processor      = processor
        self.model          = model
        self.loss_function  = loss_function

        self.num_iterations = num_iterations
        self.num_epochs     = num_epochs

        self.num_point_estimate = len(self.cfg.do_point_estimate)
        self.num_targets        = self.num_point_estimate + 1
        self.B = self.cfg.batch_size
        self.S = int(self.cfg.batch_size * self.cfg.class_balance)

        # ── Auto-configure for multibanding (mirrors SageVanillaTraining) ─
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

        device = self.cfg.device

        self.model.eval()

        diagnostics = {
            "signal_params":  [],
            "signal_idx":     [],
            "network_output": [],
            "network_target": [],
        }

        with torch.inference_mode():

            for _ in tqdm(range(self.num_iterations)):

                # ── 1. Sample ─────────────────────────────────────────────
                signal_data, signal_targets, theta = self.signal_sampler(
                    return_theta=True
                )
                noise_data, noise_targets = self.noise_sampler()

                # ── 2. Auto-multiband noise ───────────────────────────────
                if self._selector is not None:
                    noise_data = self._selector(noise_data)

                # ── 3. Pad noise targets ──────────────────────────────────
                pad = torch.zeros(
                    noise_targets.shape[0], self.num_point_estimate,
                    device=device, dtype=noise_targets.dtype,
                )
                noise_targets = torch.cat((pad, noise_targets), dim=1)

                # ── 4. Random signal injection ────────────────────────────
                idx        = torch.randperm(self.B, device=device)[: self.S]
                signal_pad = torch.zeros_like(noise_data)
                target_pad = torch.zeros(
                    self.B, self.num_targets,
                    device=device, dtype=signal_targets.dtype,
                )
                signal_pad[idx] = signal_data
                target_pad[idx] = signal_targets

                x       = noise_data + signal_pad
                targets = noise_targets + target_pad

                diagnostics["signal_idx"].append(idx.cpu())

                # ── 5. Wrap in GWBatch and preprocess ─────────────────────
                batch = GWBatch(
                    x,
                    state          = self._initial_state,
                    freqs          = self._freqs,
                    coarse_indices = self._coarse_indices,
                )
                batch     = self.processor(batch)
                net_input = batch.to_network_input()

                # ── 6. Forward pass ───────────────────────────────────────
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.cfg.autocast
                    else nullcontext()
                ):
                    out  = self.model(net_input)
                    loss = self.loss_function(out, targets)

                # ── 7. Diagnostics ────────────────────────────────────────
                network_output = torch.cat([*out], dim=1)

                ranking = network_output[:, 0:1]
                mu_std  = network_output[:, 1 : 1 + self.num_point_estimate]
                log_var = network_output[
                    :, 1 + self.num_point_estimate : 1 + 2 * self.num_point_estimate
                ]

                mu_phys = self.signal_sampler.param_sampler.unstandardise_from_batch(
                    mu_std
                )
                sigma_std  = torch.exp(0.5 * log_var)
                std_prior  = self.signal_sampler.param_sampler._std_stds.to(
                    sigma_std.device
                )
                sigma_phys = sigma_std * std_prior
                network_output = torch.cat([ranking, mu_phys, sigma_phys], dim=1)

                diagnostics["network_output"].append(network_output.cpu())
                diagnostics["network_target"].append(targets.cpu())
                diagnostics["signal_params"].append(theta.cpu())

                self.loss_components[nepoch] += loss.detach()

        self.loss_components[nepoch] /= self.num_iterations

        # ── Save diagnostic outputs ───────────────────────────────────────
        network_output = torch.vstack(diagnostics["network_output"])
        network_target = torch.vstack(diagnostics["network_target"])
        signal_params  = torch.vstack(diagnostics["signal_params"])
        signal_idx     = torch.stack(diagnostics["signal_idx"])

        savepath = os.path.join(self.cfg.export_dir, "validation_data.h5")
        save_validation(
            nepoch, network_output, network_target,
            signal_params, signal_idx, savepath,
        )
