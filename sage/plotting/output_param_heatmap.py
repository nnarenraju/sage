#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : output_param_heatmap.py
Description     : Short description of the file

Created on 2026-03-21 17:45:01

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


def plot_output_vs_param_heatmap(
    epoch,
    ranking_stat,
    labels,
    source_params,
    param_name,
    export_dir=None,
    save=True,
    bins_x=30,
    bins_y=30,
):
    """
    2D histogram heatmap of network ranking statistic vs a source parameter.

    Shows how the network output varies across the range of ``param_name``
    for both signal and noise events.  A strong vertical gradient indicates
    the parameter drives the detection score.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics.
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels.
    source_params : dict[str, array-like]
        Per-event parameter arrays.
    param_name : str
        Key of the parameter to use on one axis.
    export_dir : str or None
        Output directory.
    save : bool
        If ``True``, save to disk; otherwise display.
    bins_x : int
        Number of parameter bins (default ``30``).
    bins_y : int
        Number of ranking-statistic bins (default ``30``).
    """

    if param_name not in source_params:
        return

    param_data = source_params[param_name]

    # Only use signals
    signal_mask = labels == 1.0
    param_sig = param_data[signal_mask]
    output_sig = ranking_stat[signal_mask]

    # 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        param_sig, output_sig, bins=[bins_x, bins_y]
    )

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    plt.colorbar(label="Number of signals")
    plt.xlabel(param_name)
    plt.ylabel("Network Ranking Statistic")
    plt.title(f"Network Output vs {param_name} - Epoch {epoch}")
    plt.grid(False)

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "OUTPUT_PARAM_HEATMAP")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"{param_name}_heatmap_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
