#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : parameter_recovery.py
Description     : Short description of the file

Created on 2026-03-21 17:58:13

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
import numpy as np
import matplotlib.pyplot as plt


def plot_param_recovery_heatmap(
    all_network_outputs,  # dict epoch -> dict of param_name -> predicted
    all_labels,  # dict epoch -> dict of param_name -> true
    param_name,
    epochs,
    export_dir=None,
    save=True,
    bins=30,
):
    """
    Heatmap of median residual for a parameter across epochs
    """
    # Collect true values
    true_vals = np.concatenate([all_labels[epoch][param_name] for epoch in epochs])
    min_val, max_val = np.min(true_vals), np.max(true_vals)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    residual_matrix = []

    for epoch in epochs:
        pred = all_network_outputs[epoch][param_name]
        true = all_labels[epoch][param_name]
        residuals = pred - true

        # Compute median residual per bin
        med_res = []
        for i in range(len(bin_edges) - 1):
            idxs = np.where((true >= bin_edges[i]) & (true < bin_edges[i + 1]))[0]
            if len(idxs) > 0:
                med_res.append(np.median(residuals[idxs]))
            else:
                med_res.append(np.nan)
        residual_matrix.append(med_res)

    residual_matrix = np.array(residual_matrix)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        residual_matrix,
        aspect="auto",
        origin="lower",
        extent=[min_val, max_val, epochs[0], epochs[-1]],
        cmap="coolwarm",
    )
    plt.colorbar(label="Median Residual")
    plt.xlabel(f"True {param_name}")
    plt.ylabel("Epoch")
    plt.title(f"{param_name} Recovery Residual Heatmap Over Epochs")
    plt.grid(False)

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "PARAM_RECOVERY_HEATMAP")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"{param_name}_recovery_heatmap.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
