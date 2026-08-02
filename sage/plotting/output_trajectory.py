#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : output_trajectory.py
Description     : Evolution of the ranking-statistic distribution over epochs.

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, Sage
__license__       = GPL-3.0-or-later
__maintainer__    = Narenraju Nagarajan
"""

# Packages
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
    Ranking-statistic distribution for signals and noise as a function of epoch.

    Why this is distributional rather than per-sample
    -------------------------------------------------
    This plot used to draw one line per sample, following ``ranking_stat[i]``
    across epochs for twenty randomly chosen ``i``. That cannot work for this
    data.

    Validation batches are generated on the fly and the sampler's RNG is not
    reset between epochs, so every epoch validates on a fresh draw. Row ``i``
    in epoch 0 and row ``i`` in epoch 127 are unrelated events -- the measured
    correlation between their chirp masses is ``+0.004``, and even the
    signal/noise labels differ. The old plot joined unrelated points with
    lines, which is where the messy appearance came from.

    What the data does support is how the *distribution* moves, which is the
    more useful question anyway: are signals separating from noise, and is the
    spread tightening?

    Parameters
    ----------
    all_ranking_stats : list[array-like]
        One ranking-statistic array per epoch, ordered as ``epoch_list``.
    labels : array-like or list[array-like]
        Binary labels. A single array is applied to every epoch; pass a list of
        per-epoch label arrays, which is correct here since the draw changes.
    epoch_list : list[int]
        Epoch numbers for the x-axis.
    export_dir : str or None
        Output directory.
    save : bool
        Save to disk, else display.
    signal_idx : ignored
        Accepted for backwards compatibility. Per-sample tracking is not
        meaningful for this data; see above.
    percentiles : tuple[float]
        Percentiles to draw: median as a line, the rest as symmetric bands.
    """
    n_epochs = len(all_ranking_stats)
    if n_epochs == 0:
        return

    per_epoch_labels = (
        list(labels) if isinstance(labels, (list, tuple)) else [labels] * n_epochs
    )

    def band_stats(select_signal):
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
