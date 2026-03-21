#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : joint_cdfs.py
Description     : Short description of the file

Created on 2026-03-21 17:56:13

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


def plot_joint_cdfs(epoch, ranking_stat, labels, export_dir=None, save=True):
    """
    Plot joint CDFs of network output for signals and noise
    """
    signal_mask = labels == 1.0
    noise_mask = labels == 0.0

    sig_stats = np.sort(ranking_stat[signal_mask])
    noise_stats = np.sort(ranking_stat[noise_mask])

    cdf_sig = np.arange(1, len(sig_stats) + 1) / len(sig_stats)
    cdf_noise = np.arange(1, len(noise_stats) + 1) / len(noise_stats)

    plt.figure(figsize=(7, 6))
    plt.plot(sig_stats, cdf_sig, c="red", label="Signals")
    plt.plot(noise_stats, cdf_noise, c="blue", label="Noise")
    plt.xlabel("Network Output (Ranking Statistic)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Joint CDFs - Epoch {epoch}")
    plt.legend()
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "JOINT_CDFS")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"joint_cdfs_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
