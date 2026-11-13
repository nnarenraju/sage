#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Sage validation loop: training batches in inference mode, with each
pass's outputs, targets and parameters saved to HDF5.
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


def save_validation(name, output, target, params, param_names, signal_idx, auxiliary, savepath):
    """
    Append one validation pass's results to an HDF5 file.

    Group ``name`` holds four gzip-compressed datasets, and
    ``auxiliary`` when the run has the non-astrophysical class; an
    existing group of that name is replaced.

    Parameters
    ----------
    name : str
        Group name: ``epoch_NNNN`` for a regular pass; ``final_best``,
        ``final_ema`` or ``validate_<checkpoint>`` otherwise.
    output : torch.Tensor, shape ``(N, 1 + k * num_pe)``
        Network outputs: ranking statistic, predicted means and, with
        ``k = 2``, predicted sigmas (in physical units).
    target : torch.Tensor, shape ``(N, num_pe + 1)``
        Ground-truth targets (regression values + class label).
    params : torch.Tensor, shape ``(N, num_params)``
        Raw waveform parameters (``theta``) from the signal sampler.
    param_names : list of str
        Name of each ``params`` column, recorded as its ``names``
        attribute.
    signal_idx : torch.Tensor, shape ``(num_iter, S)``
        Batch positions where signals were injected for each iteration.
    auxiliary : torch.Tensor, shape ``(N, 1)`` or ``(N, 0)``
        The non-astrophysical vs noise logit, saved as ``auxiliary``
        when the run has the class.
    savepath : str or Path
        Path to the HDF5 output file.
    """
    # Appended in place; append_h5 moves a file damaged by a killed job
    # aside on the next open
    with append_h5(savepath) as f:
        if name in f:
            # A repeated pass replaces its group
            del f[name]
        grp = f.create_group(name)
        grp.create_dataset("network_output", data=output.numpy(), compression="gzip")
        grp.create_dataset("network_target", data=target.numpy(), compression="gzip")
        grp.create_dataset("signal_params",  data=params.numpy(),  compression="gzip")
        grp["signal_params"].attrs["names"] = param_names
        grp.create_dataset("signal_idx",     data=signal_idx.numpy(), compression="gzip")
        if auxiliary.shape[1]:
            grp.create_dataset("auxiliary", data=auxiliary.numpy(), compression="gzip")


class SageVanillaValidation(torch.nn.Module):
    """
    The Sage validation loop; one call runs one validation pass.

    Batches are made as in training and run in ``eval`` and inference
    mode; means and widths are unscaled to physical units and each
    pass's outputs, targets and parameters go to ``results_path``.

    Parameters
    ----------
    signal_sampler, noise_sampler : torch.nn.Module
        Validation samplers; the signal sampler also returns ``theta``.
    processor, model, loss_function : torch.nn.Module
        As passed to the training loop.
    num_iterations : int
        Batches per validation pass.
    num_epochs : int
        Rows of the ``loss_components`` tracking tensor.
    batch_size : int
        Windows per batch (the training one).
    signal_batch_size : int
        Windows per batch that receive a signal (the training one).
    num_estimates : int
        Estimated parameters (``signal.targets``).
    placement : torch.Generator
        Generator of the signal positions in a batch, on ``device``.
    non_astrophysical : NonAstrophysicalInjector or None
        Non-astrophysical pairs added to the noise windows; None for
        none.
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
        non_astrophysical,
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
        self.non_astrophysical = non_astrophysical

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
            "auxiliary":      [],
        }

        # Progress bar on a TTY only, as in training
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

                signal_data, signal_targets, theta = self.signal_sampler(
                    return_theta=True
                )
                noise_data, noise_targets = self.noise_sampler()

                x, targets, idx = assemble_batch(
                    signal_data, signal_targets, noise_data, noise_targets,
                    self.signal_batch_size, self.num_estimates, self.placement, self.non_astrophysical,
                )
                diagnostics["signal_idx"].append(idx.cpu())

                batch     = self.processor(GWBatch(x, state=self.processor.input_state))
                net_input = batch.to_network_input()

                with (
                    torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
                    if self.autocast_dtype is not None
                    else nullcontext()
                ):
                    out  = self.model(net_input)
                    loss = self.loss_function(out, targets)

                network_output = torch.cat([out[0], out[1]], dim=1)
                diagnostics["auxiliary"].append(out[2].float().cpu())

                ranking   = network_output[:, 0:1]
                mu_scaled = network_output[:, 1 : 1 + self.num_estimates]
                mu_phys   = self.signal_sampler.param_sampler.unscale_targets(mu_scaled)
                # Mean-only models (e.g. BCE_PE_Huber) have no width
                # columns after the means
                if network_output.shape[1] >= 1 + 2 * self.num_estimates:
                    sigma_raw = network_output[
                        :, 1 + self.num_estimates : 1 + 2 * self.num_estimates
                    ]
                    # The loss's own mapping and bounds, as in training
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

        network_output = torch.vstack(diagnostics["network_output"])
        network_target = torch.vstack(diagnostics["network_target"])
        signal_params  = torch.vstack(diagnostics["signal_params"])
        signal_idx     = torch.stack(diagnostics["signal_idx"])

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        save_validation(
            name or f"epoch_{nepoch:04d}", network_output, network_target,
            signal_params, self.signal_sampler.param_sampler.param_names, signal_idx,
            torch.vstack(diagnostics["auxiliary"]), self.results_path,
        )
