#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : perturbation_sensitivity.py
Description     : Short description of the file

Created on 2026-03-21 17:51:16

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
    Plot network output sensitivity to small perturbations in a source parameter.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Only signals
    signal_mask = labels == 1.0
    base_param = source_params[param_name][signal_mask]

    # Original network output
    base_output = ranking_stat[signal_mask]

    # Perturb the parameter +/- fraction
    perturb_outputs = []
    perturb_values = []
    for frac in np.linspace(-perturb_frac, perturb_frac, nbins):
        new_param = np.copy(base_param)
        new_param *= 1 + frac

        # Construct modified input dict
        input_dict = {k: v[signal_mask] for k, v in source_params.items()}
        input_dict[param_name] = new_param

        # Run model inference
        model_output = model.predict(input_dict)  # adjust depending on your model API
        # assume model_output[:,0] is ranking_stat
        perturb_outputs.append(model_output[:, 0])
        perturb_values.append(np.full_like(base_param, frac))

    perturb_outputs = np.concatenate(perturb_outputs)
    perturb_values = np.concatenate(perturb_values)

    plt.figure(figsize=(7, 6))
    plt.scatter(perturb_values, perturb_outputs, alpha=0.4, c="blue", s=20)
    plt.xlabel(f"Fractional perturbation in {param_name}")
    plt.ylabel("Network Ranking Statistic")
    plt.title(f"Perturbation Sensitivity - {param_name} - Epoch {epoch}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "PERTURBATION_SENSITIVITY")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"perturb_sensitivity_{param_name}_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
