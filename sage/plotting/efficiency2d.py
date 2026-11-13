#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detection efficiency of the signals in 2D bins of two source
parameters.

Filename        : efficiency2d.py
Description     : Short description of the file

Created on 2026-03-21 17:46:43

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


def plot_2d_efficiency(
    epoch,
    ranking_stat,
    labels,
    source_params,
    param_x,
    param_y,
    threshold,
    export_dir=None,
    save=True,
    min_count=20,
    bins_x=20,
    bins_y=20,
):
    """
    Plot the fraction of signals above ``threshold`` in 2D bins of two
    source parameters.

    Nothing is drawn when either parameter is absent.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Predicted ranking statistics.
    labels : array-like, shape ``(N,)``
        Class labels; signals are ``1``.
    source_params : dict[str, array-like]
        Per-signal parameter arrays.
    param_x : str
        Key in ``source_params`` for the x-axis parameter.
    param_y : str
        Key in ``source_params`` for the y-axis parameter.
    threshold : float
        Detection threshold on the ranking statistic.
    export_dir : str or None
        Saves ``{param_x}_{param_y}_efficiency_epoch_NNNN.png`` under
        ``2D_EFFICIENCY/epoch_NNNN/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    min_count : int
        Bins with fewer signals are left blank.
    bins_x : int
        Number of x-axis bins.
    bins_y : int
        Number of y-axis bins.
    """
    if param_x not in source_params or param_y not in source_params:
        return

    x = source_params[param_x][labels == 1.0]
    y = source_params[param_y][labels == 1.0]
    stat = ranking_stat[labels == 1.0]

    x_edges = np.linspace(np.min(x), np.max(x), bins_x + 1)
    y_edges = np.linspace(np.min(y), np.max(y), bins_y + 1)
    efficiency = np.zeros((bins_x, bins_y))

    for i in range(bins_x):
        for j in range(bins_y):
            idxs = np.where(
                (x >= x_edges[i])
                & (x < x_edges[i + 1])
                & (y >= y_edges[j])
                & (y < y_edges[j + 1])
            )[0]
            if len(idxs) < min_count:
                efficiency[i, j] = np.nan
            else:
                efficiency[i, j] = np.sum(stat[idxs] > threshold) / len(idxs)

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        efficiency.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, label=f"Fraction of signals above {threshold:.2f}")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title(f"2D Efficiency Map - {_etitle(epoch)}")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "2D_EFFICIENCY", _etag(epoch))
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"{param_x}_{param_y}_efficiency_{_etag(epoch)}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
