#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : epoch_separation.py
Description     : Short description of the file

Created on 2026-03-21 17:57:49

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

# Packages
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sage.plotting._epochs import epoch_numbers as _epoch_numbers


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

    fig, ax = plt.subplots(figsize=(8, 6))
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
        ax.plot(xs, kde_sig(xs),   color=colors[i], ls="-")
        ax.plot(xs, kde_noise(xs), color=colors[i], ls="--")

    # Epoch is encoded as colour, shown on a colourbar. Labelling every curve
    # instead produced a 2N-entry legend (54 for a 27-epoch run) that covered
    # half the axes and was impossible to read.
    epoch_numbers = _epoch_numbers(epochs)
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=min(epoch_numbers), vmax=max(epoch_numbers)),
    )
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Epoch")

    # Line style carries signal-vs-noise; two entries is enough.
    style_handles = [
        Line2D([], [], color="0.3", ls="-",  label="Signals"),
        Line2D([], [], color="0.3", ls="--", label="Noise"),
    ]
    ax.legend(handles=style_handles, loc="upper left", fontsize=9)

    ax.set_xlabel("Network Ranking Statistic")
    ax.set_ylabel("Density")
    ax.set_title("Signal vs Noise Separation Over Epochs")
    ax.grid(True, ls=":")

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
