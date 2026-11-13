#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fraction of signals above each noise ranking statistic, per bin of
each source parameter.

Filename        : paramfrac_above_thresh.py
Description     : Short description of the file

Created on 2026-03-21 17:39:39

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
from sage.plotting._params import select_params as _select_params


def plot_paramfrac_detected_above_thresh(
    epoch,
    ranking_stat,
    labels,
    sample_params,
    export_dir=None,
    save=True,
    nbins=4,
    params=None,
):
    """
    Plot the fraction of signals above each noise ranking statistic, per
    bin of each source parameter.

    Each noise event's ranking statistic is a threshold. Left panel: the
    parameter histogram with the bins shaded; right panel: one curve per
    bin. Nothing is drawn without both signals and noise.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the titles and output directory.
    ranking_stat : array-like, shape ``(N,)``
        Ranking statistic of every event.
    labels : array-like, shape ``(N,)``
        Class labels (1 signal, 0 noise).
    sample_params : dict[str, array-like]
        Parameter name to the value of every event.
    export_dir : str or None
        Saves ``{param}/paramfrac_{param}.png`` under
        ``PARAMFRAC_ABOVE_THRESH/epoch_NNNN/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figures.
    nbins : int
        Number of equal-width parameter bins.
    params : iterable[str] or None
        Source parameters to plot; see
        :func:`~sage.plotting._params.select_params`.
    """

    if save and export_dir is not None:
        parent_dir = os.path.join(export_dir, "PARAMFRAC_ABOVE_THRESH")
        base_dir = os.path.join(parent_dir, f"{_etag(epoch)}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    signal_mask = labels == 1.0
    noise_mask = labels == 0.0

    signal_stats = ranking_stat[signal_mask]
    noise_stats = ranking_stat[noise_mask]

    if len(signal_stats) == 0 or len(noise_stats) == 0:
        return

    sorted_noise_stats = np.sort(noise_stats)[::-1]

    colours = ["gold", "forestgreen", "orchid", "royalblue", "orangered", "gray"]

    for param, distr in _select_params(sample_params, params).items():

        masked_distr = distr[signal_mask]

        if len(masked_distr) < nbins:
            continue

        _, bin_edges = np.histogram(masked_distr, bins=nbins)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"{_etitle(epoch)}: Fraction of binned {param} above noise thresholds"
        )

        ax[0].hist(masked_distr, bins=100, histtype="step", label="Signals")
        ax[0].set_xlabel(param)
        ax[0].set_ylabel("Occurrences")
        ax[0].grid(True, ls=":")

        y_max = ax[0].get_ylim()[1]

        for n in range(len(bin_edges) - 1):

            lo = bin_edges[n]
            hi = bin_edges[n + 1]

            idxs = np.where((masked_distr > lo) & (masked_distr < hi))[0]

            if len(idxs) == 0:
                continue

            bin_stats = signal_stats[idxs]

            above = len(bin_stats) - np.searchsorted(np.sort(bin_stats), sorted_noise_stats, side="right")
            frac_detected = np.column_stack([sorted_noise_stats, above / len(bin_stats)])

            ax[0].fill_between(
                [lo, hi],
                [0, 0],
                [y_max, y_max],
                color=colours[n % len(colours)],
                alpha=0.25,
            )

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
