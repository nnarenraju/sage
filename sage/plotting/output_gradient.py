#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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


# Packages
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
    Plot the empirical gradient of network output with respect to a source parameter.

    Sorts signal events by ``param_name`` and estimates the finite-difference
    gradient of the ranking statistic with a rolling window.  A rising
    gradient indicates the network exploits this parameter; a flat curve
    indicates insensitivity.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics.
    labels : array-like, shape ``(N,)``
        Binary labels.
    source_params : dict[str, array-like]
        Per-event parameter arrays.
    param_name : str
        Key of the parameter to differentiate against.
    export_dir : str or None
        Output directory.
    save : bool
        If ``True``, save to disk; otherwise display.
    window : int
        Boxcar width used to smooth the binned trend before differentiating.
        Set to ``0`` or ``1`` to differentiate the raw binned trend.
    bins : int
        Number of parameter bins used to build the median trend (default 100).

    Notes
    -----
    The gradient is taken of the **binned median trend**, not of the raw
    per-sample values. Differentiating the samples directly divides the
    sample-to-sample scatter of the ranking statistic by the (vanishingly
    small) spacing between neighbouring parameter values, which produced
    values of order 1e4 and NaN wherever two samples shared a value.
    """

    signal_mask = labels == 1.0
    x = np.asarray(source_params[param_name])[signal_mask]
    y = np.asarray(ranking_stat)[signal_mask]

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2 or np.ptp(x) == 0:
        return

    # Differentiate the BINNED TREND, not the raw samples.
    #
    # The previous implementation did np.gradient(y_sorted, x_sorted) over every
    # signal. With 100k samples the spacing between neighbours is ~2e-4 while
    # the ranking statistic scatters by ~6 between them, so the result was
    # ~3e4 -- and outright NaN wherever two samples shared a parameter value
    # and dx was exactly 0. It measured sample noise divided by sample spacing,
    # not any gradient of the underlying relationship.
    #
    # Binning to a median trend first gives d<stat>/d<param> in units that mean
    # something (~0.5 per solar mass of chirp mass, here).
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

    # Optional smoothing of the trend before differentiating.
    if window and window > 1:
        k = int(window)
        kernel = np.ones(k) / k
        trend = np.convolve(trend, kernel, mode="same")
        # convolve tapers the ends; drop the affected edge points
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
