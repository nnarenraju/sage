#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : epoch_separation.py
Description     : Short description of the file

Created on 2026-03-21 17:57:49

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


def plot_separation_over_epochs(
    all_network_outputs,  # dict epoch -> outputs
    all_labels,  # dict epoch -> labels
    epochs,  # list of epochs
    export_dir=None,
    save=True,
):
    """
    Track signal/noise ranking-statistic separation across training epochs.

    Plots KDE curves of the ranking statistic for signal and noise samples
    for each epoch, coloured by epoch so convergence trends are visible over
    training.

    Parameters
    ----------
    all_network_outputs : dict[epoch, array-like]
        Mapping from epoch to network output arrays.
    all_labels : dict[epoch, array-like]
        Mapping from epoch to binary label arrays.
    epochs : list
        Ordered list of epoch keys to include in the plot.
    export_dir : str or None
        Output directory for the saved figure.
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    """
    import numpy as np
    from scipy.stats import gaussian_kde

    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

    for i, epoch in enumerate(epochs):
        output = all_network_outputs[epoch]
        labels = all_labels[epoch]

        sig   = output[labels == 1.0]
        noise = output[labels == 0.0]

        # KDEs via scipy
        xs = np.linspace(output.min(), output.max(), 400)
        kde_sig   = gaussian_kde(sig,   bw_method="scott")
        kde_noise = gaussian_kde(noise, bw_method="scott")
        plt.plot(xs, kde_sig(xs),   color=colors[i], ls="-",  label=f"Signals Epoch {epoch}")
        plt.plot(xs, kde_noise(xs), color=colors[i], ls="--", label=f"Noise Epoch {epoch}")

    plt.xlabel("Network Ranking Statistic")
    plt.ylabel("Density")
    plt.title("Signal vs Noise Separation Over Epochs")
    plt.legend(fontsize=8)
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "SEPARATION_OVER_EPOCHS")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, "signal_noise_separation_over_epochs.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
