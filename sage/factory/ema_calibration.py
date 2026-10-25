"""
EMA calibration at the end of training.

The per-step weight EMA averages the batch-norm running statistics along with
the weights, but the statistics that belong to the averaged weights are the
activation statistics under those weights. :func:`recalibrate_batchnorm`
recomputes them over training batches with the official
``torch.optim.swa_utils.update_bn`` (statistics reset, cumulative average).
"""

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.swa_utils import update_bn

from sage.core.pipeline import GWBatch
from sage.factory.training import assemble_batch


def calibration_inputs(signal_sampler, noise_sampler, processor, signal_batch_size, num_estimates, num_batches):
    """
    Network inputs of training batches, assembled and preprocessed as in training.

    Parameters
    ----------
    signal_sampler, noise_sampler : torch.nn.Module
        Training samplers.
    processor : Preprocessor
        Preprocessing chain.
    signal_batch_size : int
        Windows per batch that receive a signal.
    num_estimates : int
        Estimated parameters.
    num_batches : int
        Batches to yield.

    Yields
    ------
    torch.Tensor
        Network input of one batch.
    """
    for _ in range(num_batches):
        signal_data, signal_targets = signal_sampler()
        noise_data, noise_targets = noise_sampler()
        x, _, _ = assemble_batch(
            signal_data, signal_targets, noise_data, noise_targets, signal_batch_size, num_estimates,
        )
        yield processor(GWBatch(x, state=processor.input_state)).to_network_input()


def recalibrate_batchnorm(model, inputs) -> int:
    """
    Recompute the batch-norm running statistics of ``model`` from ``inputs``.

    Parameters
    ----------
    model : torch.nn.Module
        The uncompiled network holding the EMA weights.
    inputs : iterable of torch.Tensor
        Network inputs (:func:`calibration_inputs`).

    Returns
    -------
    int
        Batch-norm layers recalibrated; 0 leaves ``inputs`` undrawn.
    """
    layers = sum(isinstance(module, _BatchNorm) for module in model.modules())
    if layers:
        with torch.no_grad():
            update_bn(inputs, model)
    return layers
