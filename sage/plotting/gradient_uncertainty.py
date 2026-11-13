#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predicted uncertainty against the finite-difference gradient of the
ranking statistic in one source parameter.

Filename        : gradient_uncertainty.py
Description     : Short description of the file

Created on 2026-03-21 17:56:36

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import os
import matplotlib.pyplot as plt
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle


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
    Plot the model's uncertainty against the gradient of the ranking
    statistic in one source parameter, for the signals.

    The gradient is the forward difference with step
    ``delta_frac * value``.

    Parameters
    ----------
    model : object
        Its ``predict(params, return_uncertainty=True)`` returns
        ``(ranking_stat, uncertainty)``.
    source_params : dict[str, array-like]
        Per-event parameter arrays.
    labels : array-like, shape ``(N,)``
        Class labels; signals are ``1``.
    param_name : str
        Key of the parameter to perturb.
    export_dir : str or None
        Saves the plot under ``UNCERTAINTY_VS_GRADIENT/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    delta_frac : float
        Fractional step of the forward difference.
    epoch : int or str or None
        Epoch identifier for the title and filename.
    """
    import numpy as np

    signal_mask = labels == 1.0
    param_vals = source_params[param_name][signal_mask]

    input_dict = {k: v[signal_mask] for k, v in source_params.items()}
    ranking_stat, uncertainty = model.predict(input_dict, return_uncertainty=True)

    param_perturb = param_vals * (1 + delta_frac)
    input_dict[param_name] = param_perturb
    ranking_stat_perturb, _ = model.predict(input_dict, return_uncertainty=True)
    grad = (ranking_stat_perturb - ranking_stat) / (param_vals * delta_frac)

    plt.figure(figsize=(7, 6))
    plt.scatter(grad, uncertainty, alpha=0.5, c="purple")
    plt.xlabel(f"Gradient of Output w.r.t {param_name}")
    plt.ylabel("Predicted Uncertainty")
    plt.title(f"Uncertainty vs Gradient - {param_name} - {_etitle(epoch)}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "UNCERTAINTY_VS_GRADIENT")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(
                outdir, f"uncertainty_vs_gradient_{param_name}_{_etag(epoch)}.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
