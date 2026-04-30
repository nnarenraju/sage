#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : cumulative_volume.py
Description     : Short description of the file

Created on 2026-03-21 17:49:02

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


def plot_cumulative_volume(
    epoch,
    ranking_stat,
    labels,
    source_params,
    distance_param="distance",
    export_dir=None,
    save=True,
    bins=50,
):
    """
    Plot cumulative detection volume as a function of network output.
    """

    # Only signals
    signal_mask = labels == 1.0
    output_sig = ranking_stat[signal_mask]
    distances = source_params[distance_param][signal_mask]

    # Compute volume weights
    volumes = distances**3  # proxy for volume

    # Sort by network output descending
    sort_idx = np.argsort(output_sig)[::-1]
    output_sorted = output_sig[sort_idx]
    vol_sorted = volumes[sort_idx]

    # cumulative sum
    cum_vol = np.cumsum(vol_sorted)
    cum_vol /= cum_vol[-1]  # normalize to 1

    plt.figure(figsize=(7, 6))
    plt.plot(output_sorted, cum_vol, c="blue", lw=2)
    plt.xlabel("Network Ranking Statistic")
    plt.ylabel("Cumulative Fraction of Accessible Volume")
    plt.title(f"Cumulative Detection Volume - Epoch {epoch}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "CUMULATIVE_VOLUME")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"cumulative_volume_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
