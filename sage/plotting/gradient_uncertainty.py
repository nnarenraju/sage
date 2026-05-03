#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : gradient_uncertainty.py
Description     : Short description of the file

Created on 2026-03-21 17:56:36

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
import matplotlib.pyplot as plt


def plot_uncertainty_vs_gradient(
    model,
    source_params,
    labels,
    param_name,
    export_dir=None,
    save=True,
    delta_frac=0.01,
    epoch=None,
):
    """
    Scatter plot of model uncertainty vs finite-difference output gradient.

    Perturbs ``param_name`` by ``delta_frac × value`` and computes the
    numerical gradient of the ranking statistic.  Plots this against the
    heteroscedastic uncertainty predicted by the model for signal events.
    Ideally, high gradient should co-occur with high uncertainty.

    Parameters
    ----------
    model : object with ``predict`` method
        Model that returns ``(ranking_stat, uncertainty)`` when called with
        ``return_uncertainty=True``.
    source_params : dict[str, array-like]
        Per-event parameter arrays.
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels.
    param_name : str
        Key of the parameter to perturb.
    export_dir : str or None
        Output directory.
    save : bool
        If ``True``, save to disk; otherwise display.
    delta_frac : float
        Fractional perturbation size for the gradient estimate (default ``0.01``).
    epoch : int or str or None
        Epoch identifier for the filename.
    """
    import numpy as np

    signal_mask = labels == 1.0
    param_vals = source_params[param_name][signal_mask]

    # Original outputs
    input_dict = {k: v[signal_mask] for k, v in source_params.items()}
    ranking_stat, uncertainty = model.predict(input_dict, return_uncertainty=True)

    # Gradient approx
    param_perturb = param_vals * (1 + delta_frac)
    input_dict[param_name] = param_perturb
    ranking_stat_perturb, _ = model.predict(input_dict, return_uncertainty=True)
    grad = (ranking_stat_perturb - ranking_stat) / (param_vals * delta_frac)

    plt.figure(figsize=(7, 6))
    plt.scatter(grad, uncertainty, alpha=0.5, c="purple")
    plt.xlabel(f"Gradient of Output w.r.t {param_name}")
    plt.ylabel("Predicted Uncertainty")
    plt.title(f"Uncertainty vs Gradient - {param_name} - Epoch {epoch}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "UNCERTAINTY_VS_GRADIENT")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(
                outdir, f"uncertainty_vs_gradient_{param_name}_epoch_{epoch}.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
