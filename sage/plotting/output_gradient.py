#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : output_gradient.py
Description     : Short description of the file

Created on 2026-03-21 17:49:22

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
def plot_output_gradient(
    epoch,
    ranking_stat,
    labels,
    source_params,
    param_name,
    export_dir=None,
    save=True,
    window=5,
):
    """
    Plot network output gradient w.r.t. a source parameter
    """

    signal_mask = labels == 1.0
    x = source_params[param_name][signal_mask]
    y = ranking_stat[signal_mask]

    # sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # rolling derivative
    dy_dx = np.gradient(y_sorted, x_sorted)

    plt.figure(figsize=(7, 6))
    plt.plot(x_sorted, dy_dx, lw=2, c="green")
    plt.xlabel(param_name)
    plt.ylabel("d(Network Output)/d(param)")
    plt.title(f"Output Gradient vs {param_name} - Epoch {epoch}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "OUTPUT_GRADIENT")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"output_gradient_{param_name}_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
