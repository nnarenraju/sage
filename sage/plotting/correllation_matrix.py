#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : correllation_matrix.py
Description     : Short description of the file

Created on 2026-03-21 17:52:33

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


def plot_correlation_matrix(
    ranking_stat, source_params, labels, export_dir=None, save=True, epoch=None
):
    """
    Plot a Pearson correlation matrix between source parameters and ranking statistic.

    Computes pairwise correlations for all source parameters plus the ranking
    statistic, restricting to signal events and dropping any rows containing
    NaN values.  Useful for checking whether the network's output is driven
    by a single parameter.

    Parameters
    ----------
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics.
    source_params : dict[str, array-like]
        Per-event source parameter arrays.
    labels : array-like, shape ``(N,)``
        Binary labels; only signal rows (label == 1) are included.
    export_dir : str or None
        Output directory.
    save : bool
        If ``True``, save to disk; otherwise display.
    epoch : int or str or None
        Epoch identifier for the filename (optional).
    """
    import numpy as np

    # Only signals (drop NaN rows from unaligned noise entries)
    signal_mask = labels == 1.0
    keys = list(source_params.keys()) + ["ranking_stat"]
    cols = [source_params[k][signal_mask] for k in source_params] + [ranking_stat[signal_mask]]

    # Stack and drop rows with NaN
    mat = np.column_stack(cols)
    valid = np.all(np.isfinite(mat), axis=1)
    mat = mat[valid]

    corr = np.corrcoef(mat.T)

    fig, ax = plt.subplots(figsize=(max(6, len(keys)), max(5, len(keys) - 1)))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(f"Correlation Matrix - Epoch {epoch}")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "CORRELATION_MATRIX")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"correlation_matrix_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
