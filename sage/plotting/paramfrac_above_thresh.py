#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : paramfrac_above_thresh.py
Description     : Short description of the file

Created on 2026-03-21 17:39:39

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


def plot_paramfrac_detected_above_thresh(
    epoch,
    ranking_stat,
    labels,
    sample_params,
    export_dir=None,
    save=True,
    nbins=4,
):
    """
    Detection fraction above threshold in bins of each source parameter.

    For each parameter, divides signals into ``nbins`` equal-count bins and
    plots the fraction detected (ranking statistic above a sweep of
    thresholds) in each bin.  Reveals which parameter values the model
    finds hardest to detect.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier used in the output subdirectory.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics.
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels.
    sample_params : dict[str, array-like]
        Per-event parameter arrays.
    export_dir : str or None
        Parent directory; plots saved under
        ``PARAMFRAC_ABOVE_THRESH/epoch_{epoch}/``.
    save : bool
        If ``True``, save to disk; otherwise display.
    nbins : int
        Number of equal-count parameter bins (default ``4``).
    """

    if save and export_dir is not None:
        parent_dir = os.path.join(export_dir, "PARAMFRAC_ABOVE_THRESH")
        base_dir = os.path.join(parent_dir, f"epoch_{epoch}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    # --------------------------------------------
    # masks
    # --------------------------------------------
    signal_mask = labels == 1.0
    noise_mask = labels == 0.0

    signal_stats = ranking_stat[signal_mask]
    noise_stats = ranking_stat[noise_mask]

    if len(signal_stats) == 0 or len(noise_stats) == 0:
        return

    sorted_noise_stats = np.sort(noise_stats)[::-1]

    colours = ["gold", "forestgreen", "orchid", "royalblue", "orangered", "gray"]

    # --------------------------------------------
    # loop parameters
    # --------------------------------------------
    for param, distr in sample_params.items():

        masked_distr = distr[signal_mask]

        if len(masked_distr) < nbins:
            continue

        # parameter bins
        _, bin_edges = np.histogram(masked_distr, bins=nbins)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Epoch {epoch}: Fraction of binned {param} above noise thresholds"
        )

        # overall signal distribution
        ax[0].hist(masked_distr, bins=100, histtype="step", label="Signals")
        ax[0].set_xlabel(param)
        ax[0].set_ylabel("Occurrences")
        ax[0].grid(True, ls=":")

        y_max = ax[0].get_ylim()[1]

        # --------------------------------------------
        # per-bin efficiency curves
        # --------------------------------------------
        for n in range(len(bin_edges) - 1):

            lo = bin_edges[n]
            hi = bin_edges[n + 1]

            idxs = np.where((masked_distr > lo) & (masked_distr < hi))[0]

            if len(idxs) == 0:
                continue

            bin_stats = signal_stats[idxs]

            frac_detected = []

            for thresh in sorted_noise_stats:
                frac = np.sum(bin_stats > thresh) / len(bin_stats)
                frac_detected.append([thresh, frac])

            frac_detected = np.array(frac_detected)

            # shade parameter bin
            ax[0].fill_between(
                [lo, hi],
                [0, 0],
                [y_max, y_max],
                color=colours[n % len(colours)],
                alpha=0.25,
            )

            # efficiency curve
            ax[1].plot(
                frac_detected[:, 0],
                frac_detected[:, 1],
                color=colours[n % len(colours)],
                label=f"{lo:.2f} → {hi:.2f}",
            )

        ax[1].set_xlabel("Noise Stat Threshold")
        ax[1].set_ylabel("Frac Signals Detected")
        ax[1].grid(True, ls=":")
        ax[1].legend()

        if save and base_dir is not None:
            out_dir = os.path.join(base_dir, param)
            os.makedirs(out_dir, exist_ok=True)

            fig.savefig(
                os.path.join(out_dir, f"paramfrac_{param}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)
