#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mean ranking statistic of the signals in bins of network SNR.

Filename        : snr_confidence.py
Description     : Short description of the file

Created on 2026-03-21 17:55:24

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


def plot_confidence_vs_snr(
    epoch, ranking_stat, labels, network_snrs, export_dir=None, save=True, snr_bins=20
):
    """
    Plot the mean and standard deviation of the signals' ranking
    statistic in bins of network SNR.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Ranking statistic of every event.
    labels : array-like, shape ``(N,)``
        Class labels; signals are ``1``.
    network_snrs : array-like, shape ``(N,)``
        Network SNR of every event.
    export_dir : str or None
        Saves the plot under ``CONFIDENCE_VS_SNR/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    snr_bins : int
        Number of equal-width SNR bins.
    """
    signal_mask = labels == 1.0
    snrs = network_snrs[signal_mask]
    stats = ranking_stat[signal_mask]

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
    plt.title(f"Confidence vs SNR - {_etitle(epoch)}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "CONFIDENCE_VS_SNR")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"confidence_vs_snr_{_etag(epoch)}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
