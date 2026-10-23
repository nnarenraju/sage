#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : prediction.py
Description     : Short description of the file

Created on 2026-03-21 17:26:02

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
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle


def plot_prediction_raw(epoch, ranking_stat, labels, export_dir=None, save=True):
    """
    Plot raw ranking-statistic distributions and a FAR-efficiency sweep.

    Produces two panels: (1) overlapping histograms of the raw ranking
    statistic for signal and noise events, and (2) a detection-efficiency-
    vs-threshold curve that shows the signal fraction recovered as a function
    of the noise false-alarm rate.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics for all validation events.
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels (1 = signal, 0 = noise).
    export_dir : str or None
        Output directory (saved under ``PRED_RAW/``).
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    """

    save_dir = None
    if save and export_dir is not None:
        save_dir = os.path.join(export_dir, "PRED_RAW")
        os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------
    # Split signal / noise
    # --------------------------------------------
    signal_mask = labels == 1.0
    noise_mask = labels == 0.0

    raw_tp = ranking_stat[signal_mask]
    raw_tn = ranking_stat[noise_mask]

    # --------------------------------------------
    # FAR sweep curve (efficiency vs noise threshold)
    # --------------------------------------------
    sorted_noise_stats = np.sort(raw_tn)

    frac_detected = []

    for thresh in sorted_noise_stats[::-1]:
        frac = np.sum(raw_tp > thresh) / len(raw_tp)
        frac_detected.append([thresh, frac])

    frac_detected = np.array(frac_detected)

    # --------------------------------------------
    # Plot
    # --------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Raw Output at {_etitle(epoch)}")

    # Histogram panel
    ax[0].hist(raw_tp, bins=100, histtype="step", label="Signals")
    ax[0].hist(raw_tn, bins=100, histtype="step", label="Noise")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Ranking Statistic")
    ax[0].set_ylabel("Number of Occurrences")
    ax[0].legend()
    ax[0].grid(True, ls=":")

    # Efficiency panel
    ax[1].plot(
        frac_detected[:, 0],
        frac_detected[:, 1],
        color="red",
        label=f"Top FAR bin: {int(frac_detected[0,1] * len(raw_tp))}/{len(raw_tp)}",
    )

    ax[1].set_xlabel("Noise Stat Threshold")
    ax[1].set_ylabel("Frac Signals Detected above Threshold")
    ax[1].set_xlim(np.min(frac_detected[:, 0]), np.max(frac_detected[:, 0]))
    ax[1].grid(True, ls=":")
    ax[1].legend()

    # Secondary axis → number of signals
    convert = lambda frac: frac * len(raw_tp)
    inverse = lambda num: num / len(raw_tp)

    secay = ax[1].secondary_yaxis("right", functions=(convert, inverse))
    secay.set_ylabel("Num Signals Detected above Threshold")

    plt.tight_layout()

    if save and save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, f"log_raw_output_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
