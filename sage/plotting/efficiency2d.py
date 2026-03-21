#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : efficiency2d.py
Description     : Short description of the file

Created on 2026-03-21 17:46:43

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
    bins_x=20,
    bins_y=20,
):
    """
    2D efficiency map: fraction of signals above threshold per parameter bin
    """
    if param_x not in source_params or param_y not in source_params:
        return

    x = source_params[param_x][labels == 1.0]
    y = source_params[param_y][labels == 1.0]
    stat = ranking_stat[labels == 1.0]

    # 2D bins
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
            if len(idxs) == 0:
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
    plt.title(f"2D Efficiency Map - Epoch {epoch}")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "2D_EFFICIENCY")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"{param_x}_{param_y}_efficiency_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
