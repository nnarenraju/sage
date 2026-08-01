#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : param_distribution.py
Description     : Short description of the file

Created on 2026-03-21 17:38:07

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
import itertools


def plot_outputbin_param_distribution(
    epoch,
    ranking_stat,
    labels,
    sample_params,
    export_dir=None,
    save=True,
):
    """
    Distribution of source parameters within ranking-statistic output bins.

    Bins all events by their ranking statistic and plots the parameter
    distribution (histogram) within each bin.  Highlights whether certain
    parameter values cluster at low or high confidence scores.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier used in the output subdirectory.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics.
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels.
    sample_params : dict[str, array-like]
        Per-event source parameter arrays.
    export_dir : str or None
        Parent directory; plots saved under ``OUTBIN_PARAM_DISTR/epoch_{epoch}/``.
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    """

    if save and export_dir is not None:
        parent_dir = os.path.join(export_dir, "OUTBIN_PARAM_DISTR")
        base_dir = os.path.join(parent_dir, f"epoch_{epoch}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    # --------------------------------------------
    # split signal / noise
    # --------------------------------------------
    signal_mask = labels == 1.0
    noise_mask = labels == 0.0

    signal_stats = ranking_stat[signal_mask]
    noise_stats = ranking_stat[noise_mask]

    if len(signal_stats) == 0 or len(noise_stats) == 0:
        return

    # --------------------------------------------
    # define experimental bin ranges
    # --------------------------------------------
    min_sig = np.min(signal_stats)
    max_sig = np.max(signal_stats)
    max_noise = np.max(noise_stats)

    high_sig = signal_stats[signal_stats > max_noise]

    if len(high_sig) > 0:
        med_high_sig = np.median(high_sig)
    else:
        med_high_sig = max_sig

    bin_ranges = [
        (min_sig, 0.0),
        (min_sig, max_noise),
        (0.0, max_noise),
        (max_noise, max_sig),
        (max_noise, med_high_sig),
        (med_high_sig, max_sig),
        (0.0, max_sig),
    ]

    bin_names = [
        "min_signal → 0",
        "min_signal → max_noise",
        "0 → max_noise",
        "max_noise → max_signal",
        "max_noise → median(high_signal)",
        "median(high_signal) → max_signal",
        "0 → max_signal",
    ]

    # histogram style
    hist_kwargs = dict(histtype="stepfilled", alpha=0.5, bins=100)

    # --------------------------------------------
    # loop bins
    # --------------------------------------------
    for n, (lo, hi) in enumerate(bin_ranges):

        idxs = np.where((signal_stats > lo) & (signal_stats < hi))[0]

        if len(idxs) == 0:
            continue

        if save and base_dir is not None:
            outdir = os.path.join(base_dir, f"bin_range_{n}")
            os.makedirs(outdir, exist_ok=True)
        else:
            outdir = None

        # --------------------------------------------
        # parameter distribution panel
        # --------------------------------------------
        params = list(sample_params.keys())
        ncols = 3
        nrows = int(np.ceil(len(params) / ncols))

        fig, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        ax = np.atleast_2d(ax)

        grid_positions = list(itertools.product(range(nrows), range(ncols)))

        for (param, distr), (i, j) in zip(sample_params.items(), grid_positions):

            distr_sig = distr[signal_mask]

            ax[i, j].hist(distr_sig[idxs], label="binned", color="blue", **hist_kwargs)
            ax[i, j].hist(distr_sig, label="all", color="red", **hist_kwargs)

            ax[i, j].set_title(param)
            ax[i, j].grid(True, ls=":")
            ax[i, j].legend()

        # hide empty panels
        for i, j in grid_positions[len(params) :]:
            ax[i, j].set_visible(False)

        plt.tight_layout()

        if save and outdir is not None:
            fig.savefig(
                os.path.join(outdir, "outbin_distr.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

        # --------------------------------------------
        # ranking stat histogram with bin highlight
        # --------------------------------------------
        plt.figure(figsize=(7, 6))

        plt.hist(signal_stats, bins=100, histtype="step", label="Signals")
        plt.yscale("log")

        plt.axvline(lo, color="k", ls=":", lw=2, label=f"lower = {lo:.2f}")
        plt.axvline(hi, color="k", ls="--", lw=2, label=f"upper = {hi:.2f}")

        plt.xlabel("Network Raw Output")
        plt.ylabel("Occurrences")
        plt.title(bin_names[n])
        plt.grid(True, ls=":")
        plt.legend()

        if save and outdir is not None:
            plt.savefig(
                os.path.join(outdir, "binned_network_output.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()
            plt.close()
