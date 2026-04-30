#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : output_trajectory.py
Description     : Short description of the file

Created on 2026-03-21 17:55:48

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


def plot_output_trajectory_over_epochs(
    all_ranking_stats, labels, epoch_list, export_dir=None, save=True, signal_idx=None
):
    """
    Plot network output trajectory for a selection of signals over epochs
    all_ranking_stats: list of arrays per epoch
    signal_idx: optional list of indices to plot
    """
    # Signals only
    signal_mask = labels == 1.0
    n_signals = np.sum(signal_mask)

    if signal_idx is None:
        signal_idx = np.random.choice(
            np.where(signal_mask)[0], min(20, n_signals), replace=False
        )

    plt.figure(figsize=(8, 6))
    for idx in signal_idx:
        y = [all_ranking_stats[ep][idx] for ep in range(len(epoch_list))]
        plt.plot(epoch_list, y, "-o", alpha=0.7)

    plt.xlabel("Epoch")
    plt.ylabel("Network Output (Ranking Statistic)")
    plt.title("Network Output Trajectory for Selected Signals")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "OUTPUT_TRAJECTORY")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"output_trajectory_signals.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
