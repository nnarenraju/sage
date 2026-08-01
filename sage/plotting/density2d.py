#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : density2d.py
Description     : Short description of the file

Created on 2026-03-21 17:47:18

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


def plot_2d_param_density(
    epoch,
    ranking_stat,
    labels,
    source_params,
    param_x,
    param_y,
    export_dir=None,
    save=True,
    bins=50,
):
    """
    Scatter/hexbin of two source parameters coloured by ranking statistic.

    Visualises how the network's confidence is distributed across the 2D
    parameter space spanned by ``param_x`` and ``param_y`` for signal events.
    Reveals parameter degeneracies or sensitivity gradients.

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
    param_x : str
        Key for the x-axis parameter.
    param_y : str
        Key for the y-axis parameter.
    export_dir : str or None
        Output directory.
    save : bool
        If ``True``, save to disk; otherwise display.
    bins : int
        Number of hexbin bins (default ``50``).
    """
    if param_x not in source_params or param_y not in source_params:
        return

    x = source_params[param_x][labels == 1.0]
    y = source_params[param_y][labels == 1.0]
    z = ranking_stat[labels == 1.0]

    plt.figure(figsize=(7, 6))
    h = plt.hist2d(x, y, bins=bins, weights=z, cmap="plasma")
    plt.colorbar(label="Sum of ranking statistic")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.title(f"Ranking Statistic Density - Epoch {epoch}")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "2D_PARAM_DENSITY")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(
                outdir, f"{param_x}_{param_y}_ranking_density_epoch_{epoch}.png"
            ),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
