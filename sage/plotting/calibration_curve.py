#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibration curve: observed signal fraction per bin of the ranking
statistic.

Filename        : calibration_curve.py
Description     : Short description of the file

Created on 2026-03-21 17:43:40

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


def plot_calibration_curve(
    epoch,
    ranking_stat,
    labels,
    export_dir=None,
    save=True,
    nbins=20,
):
    """
    Plot the fraction of signals in each bin of the ranking statistic.

    Markers sit at the bin centres, sized and coloured by the bin count,
    against a line from (min, 0) to (max, 1).

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Predicted ranking statistics.
    labels : array-like, shape ``(N,)``
        Class labels (1 signal, 0 noise); other classes are left out.
    export_dir : str or None
        Saves ``calibration/calibration_epoch_NNNN.png`` under it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    nbins : int
        Number of equal-width bins between the extremes.
    """
    keep = (labels == 0.0) | (labels == 1.0)
    ranking_stat, labels = ranking_stat[keep], labels[keep]
    signal_mask = labels == 1.0
    signal_stats = ranking_stat[signal_mask]

    bin_edges = np.linspace(np.min(ranking_stat), np.max(ranking_stat), nbins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    frac_observed = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        idxs = np.where((ranking_stat >= lo) & (ranking_stat < hi))[0]
        bin_counts.append(len(idxs))
        if len(idxs) == 0:
            frac_observed.append(np.nan)
        else:
            frac_observed.append(np.sum(labels[idxs] == 1.0) / len(idxs))

    frac_observed = np.array(frac_observed)
    bin_counts = np.array(bin_counts)

    plt.figure(figsize=(7, 6))
    plt.plot(
        [np.min(ranking_stat), np.max(ranking_stat)],
        [0, 1],
        ls="--",
        color="k",
        label="Perfect Calibration",
    )
    sc = plt.scatter(
        bin_centers,
        frac_observed,
        s=bin_counts / 100.0,
        c=bin_counts,
        cmap="viridis",
    )
    plt.xlim(np.min(ranking_stat), np.max(ranking_stat))
    plt.ylim(0, 1)
    plt.colorbar(sc, label="Number of Samples in Bin")
    plt.xlabel("Predicted Ranking Statistic")
    plt.ylabel("Observed Fraction of Signals")
    plt.title(f"Calibration Curve - {_etitle(epoch)}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "calibration")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"calibration_{_etag(epoch)}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
