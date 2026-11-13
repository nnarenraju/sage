#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Percentiles of the ranking statistic of signals and noise across
epochs.

Filename        : output_trajectory.py
Description     : Evolution of the ranking-statistic distribution over epochs.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_output_trajectory_over_epochs(
    all_ranking_stats,
    labels,
    epoch_list,
    export_dir=None,
    save=True,
    signal_idx=None,
    percentiles=(10, 25, 50, 75, 90),
):
    """
    Plot percentiles of the ranking statistic of signals and noise
    against epoch.

    Each epoch's events are a separate draw, so the plot follows the
    distribution rather than individual events. Nothing is drawn when
    there are no epochs.

    Parameters
    ----------
    all_ranking_stats : list[array-like]
        One ranking-statistic array per epoch, in ``epoch_list`` order.
    labels : array-like or list[array-like]
        Class labels (1 signal, 0 noise): a list or tuple of one array
        per epoch, or one array used for every epoch.
    epoch_list : list[int]
        Epoch numbers for the x-axis.
    export_dir : str or None
        Saves ``OUTPUT_TRAJECTORY/output_trajectory_distribution.png``
        under it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    signal_idx : object
        Not used.
    percentiles : tuple[float]
        The middle one is drawn as a line; the others pair from the
        outside in as bands.
    """
    n_epochs = len(all_ranking_stats)
    if n_epochs == 0:
        return

    per_epoch_labels = (
        list(labels) if isinstance(labels, (list, tuple)) else [labels] * n_epochs
    )

    def band_stats(select_signal):
        """Percentiles per epoch, shape ``(E, len(percentiles))``."""
        rows = []
        for stats, lab in zip(all_ranking_stats, per_epoch_labels):
            stats = np.asarray(stats)
            lab = np.asarray(lab)
            sel = (lab == 1.0) if select_signal else (lab == 0.0)
            vals = stats[sel]
            rows.append(
                np.percentile(vals, percentiles)
                if vals.size else np.full(len(percentiles), np.nan)
            )
        return np.asarray(rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    mid = len(percentiles) // 2

    for select_signal, colour, name in (
        (True, "tab:blue", "Signals"),
        (False, "tab:orange", "Noise"),
    ):
        q = band_stats(select_signal)
        for lo in range(mid):
            hi = len(percentiles) - 1 - lo
            ax.fill_between(
                epoch_list, q[:, lo], q[:, hi],
                color=colour, alpha=0.15 + 0.12 * lo, linewidth=0,
            )
        ax.plot(epoch_list, q[:, mid], color=colour, lw=2, label=f"{name} (median)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Network Output (Ranking Statistic)")
    ax.set_title(
        "Ranking-Statistic Distribution Over Epochs\n"
        f"median with {percentiles[0]}-{percentiles[-1]}th percentile bands"
    )
    ax.grid(True, ls=":")
    ax.legend()

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "OUTPUT_TRAJECTORY")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, "output_trajectory_distribution.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
