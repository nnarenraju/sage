#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradient of the binned median ranking statistic of the signals in
one source parameter.

Filename        : output_gradient.py
Description     : Short description of the file

Created on 2026-03-21 17:49:22

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


def plot_output_gradient(
    epoch,
    ranking_stat,
    labels,
    source_params,
    param_name,
    export_dir=None,
    save=True,
    window=5,
    bins=100,
):
    """
    Plot the gradient of the signals' median ranking statistic in one
    source parameter.

    The median per parameter bin is boxcar-smoothed, then differentiated
    with ``numpy.gradient``. Nothing is drawn when fewer than 3 bins are
    populated or the parameter has no spread.

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
        Key of the parameter to differentiate against.
    export_dir : str or None
        Saves the plot under ``OUTPUT_GRADIENT/epoch_NNNN/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    window : int
        Boxcar width in bins; 0 or 1 leaves the medians unsmoothed.
    bins : int
        Number of equal-width parameter bins.
    """

    signal_mask = labels == 1.0
    x = np.asarray(source_params[param_name])[signal_mask]
    y = np.asarray(ranking_stat)[signal_mask]

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.ptp(x) == 0:
        return

    # Median per bin; differentiating raw samples divides scatter by
    # the spacing of neighbouring values
    edges = np.linspace(x.min(), x.max(), bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)

    trend = np.full(bins, np.nan)
    for b in range(bins):
        sel = which == b
        if sel.any():
            trend[b] = np.median(y[sel])

    populated = np.isfinite(trend)
    if populated.sum() < 3:
        return
    centres, trend = centres[populated], trend[populated]

    if window and window > 1:
        k = int(window)
        kernel = np.ones(k) / k
        trend = np.convolve(trend, kernel, mode="same")
        # Drop the edge points tapered by the convolution
        edge = k // 2
        if len(trend) > 2 * edge + 2 and edge > 0:
            centres, trend = centres[edge:-edge], trend[edge:-edge]

    x_sorted = centres
    dy_dx = np.gradient(trend, centres)

    plt.figure(figsize=(7, 6))
    plt.plot(x_sorted, dy_dx, lw=2, c="green")
    plt.xlabel(param_name)
    plt.ylabel("d(Network Output)/d(param)")
    plt.title(f"Output Gradient vs {param_name} - {_etitle(epoch)}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "OUTPUT_GRADIENT", _etag(epoch))
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"output_gradient_{param_name}_{_etag(epoch)}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
