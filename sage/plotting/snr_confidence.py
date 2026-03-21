#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : snr_confidence.py
Description     : Short description of the file

Created on 2026-03-21 17:55:24

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


def plot_confidence_vs_snr(
    epoch, ranking_stat, labels, network_snrs, export_dir=None, save=True, snr_bins=20
):
    """
    Plot average network output (confidence) in bins of SNR for signals
    """
    signal_mask = labels == 1.0
    snrs = network_snrs[signal_mask]
    stats = ranking_stat[signal_mask]

    # Bin edges
    bins = np.linspace(np.min(snrs), np.max(snrs), snr_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    avg_conf = np.zeros_like(bin_centers)
    std_conf = np.zeros_like(bin_centers)

    for i in range(len(bin_centers)):
        idxs = np.where((snrs >= bins[i]) & (snrs < bins[i + 1]))[0]
        if len(idxs) > 0:
            avg_conf[i] = np.mean(stats[idxs])
            std_conf[i] = np.std(stats[idxs])
        else:
            avg_conf[i] = np.nan
            std_conf[i] = np.nan

    plt.figure(figsize=(7, 6))
    plt.errorbar(bin_centers, avg_conf, yerr=std_conf, fmt="o-", c="blue")
    plt.xlabel("Network SNR")
    plt.ylabel("Mean Network Output (Confidence)")
    plt.title(f"Confidence vs SNR - Epoch {epoch}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "CONFIDENCE_VS_SNR")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"confidence_vs_snr_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
