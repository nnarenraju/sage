#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D histogram of the ranking statistic of the signals against one
source parameter.

Filename        : output_param_heatmap.py
Description     : Short description of the file

Created on 2026-03-21 17:45:01

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
import numpy as np
import matplotlib.pyplot as plt
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle


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
    Plot the 2D histogram of the signals' ranking statistic against one
    source parameter.

    Nothing is drawn when the parameter is absent.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics.
    labels : array-like, shape ``(N,)``
        Class labels; signals are ``1``.
    source_params : dict[str, array-like]
        Per-event parameter arrays.
    param_name : str
        Key of the x-axis parameter.
    export_dir : str or None
        Saves the plot under ``OUTPUT_PARAM_HEATMAP/epoch_NNNN/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    bins_x : int
        Number of parameter bins.
    bins_y : int
        Number of ranking-statistic bins.
    """

    if param_name not in source_params:
        return

    param_data = source_params[param_name]

    signal_mask = labels == 1.0
    param_sig = param_data[signal_mask]
    output_sig = ranking_stat[signal_mask]

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
    plt.title(f"Network Output vs {param_name} - {_etitle(epoch)}")
    plt.grid(False)

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "OUTPUT_PARAM_HEATMAP", _etag(epoch))
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"{param_name}_heatmap_{_etag(epoch)}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
