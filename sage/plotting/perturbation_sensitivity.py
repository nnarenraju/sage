#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ranking statistic of the signals under fractional perturbations of
one source parameter.

Filename        : perturbation_sensitivity.py
Description     : Short description of the file

Created on 2026-03-21 17:51:16

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
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle


def plot_perturbation_sensitivity(
    model,
    ranking_stat,
    labels,
    source_params,
    param_name,
    export_dir=None,
    save=True,
    perturb_frac=0.05,
    nbins=20,
    epoch=None,
):
    """
    Plot the signals' ranking statistic against a fractional
    perturbation of one source parameter.

    ``param_name`` is scaled by ``1 + frac`` for ``nbins`` values of
    ``frac`` evenly spaced in ``[-perturb_frac, perturb_frac]``.

    Parameters
    ----------
    model : object
        Its ``predict(params)`` returns an array whose column 0 is the
        ranking statistic.
    ranking_stat : array-like, shape ``(N,)``
        Ranking statistic of every event; not drawn.
    labels : array-like, shape ``(N,)``
        Class labels; signals are ``1``.
    source_params : dict[str, array-like]
        Per-event parameter arrays.
    param_name : str
        Key of the parameter to perturb.
    export_dir : str or None
        Saves the plot under ``PERTURBATION_SENSITIVITY/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    perturb_frac : float
        Largest fractional perturbation.
    nbins : int
        Number of perturbation values.
    epoch : int or str or None
        Epoch identifier for the title and filename.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    signal_mask = labels == 1.0
    base_param = source_params[param_name][signal_mask]

    base_output = ranking_stat[signal_mask]

    perturb_outputs = []
    perturb_values = []
    for frac in np.linspace(-perturb_frac, perturb_frac, nbins):
        new_param = np.copy(base_param)
        new_param *= 1 + frac

        input_dict = {k: v[signal_mask] for k, v in source_params.items()}
        input_dict[param_name] = new_param

        model_output = model.predict(input_dict)
        # Column 0 is the ranking statistic
        perturb_outputs.append(model_output[:, 0])
        perturb_values.append(np.full_like(base_param, frac))

    perturb_outputs = np.concatenate(perturb_outputs)
    perturb_values = np.concatenate(perturb_values)

    plt.figure(figsize=(7, 6))
    plt.scatter(perturb_values, perturb_outputs, alpha=0.4, c="blue", s=20)
    plt.xlabel(f"Fractional perturbation in {param_name}")
    plt.ylabel("Network Ranking Statistic")
    plt.title(f"Perturbation Sensitivity - {param_name} - {_etitle(epoch)}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "PERTURBATION_SENSITIVITY")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"perturb_sensitivity_{param_name}_{_etag(epoch)}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
