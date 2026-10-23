#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : prediction_probability.py
Description     : Short description of the file

Created on 2026-03-21 17:27:00

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


def plot_prediction_probability(epoch, pred_prob, labels, export_dir=None, save=True):
    """
    Plot overlapping histograms of predicted ranking statistic for signal and noise.

    Visualises how well the network separates the two classes by plotting
    the distribution of ``pred_prob`` for label=1 (signal) and label=0 (noise)
    on the same axes.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    pred_prob : array-like, shape ``(N,)``
        Predicted ranking statistics (raw logits or probabilities).
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels (1 = signal, 0 = noise).
    export_dir : str or None
        Directory to save the figure (under ``PRED_PROB/``).
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    """

    save_dir = None
    if save and export_dir is not None:
        save_dir = os.path.join(export_dir, "PRED_PROB")
        os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------
    # Split signal / noise
    # --------------------------------------------
    signal_mask = labels == 1.0
    noise_mask = labels == 0.0

    pred_prob_tp = pred_prob[signal_mask]
    pred_prob_tn = pred_prob[noise_mask]

    # Diagnostic separation metric (not returned)
    boundary_diff = np.round(np.max(pred_prob_tp) - np.max(pred_prob_tn), 8)

    # --------------------------------------------
    # Plot
    # --------------------------------------------
    plt.figure(figsize=(8, 6))

    plt.hist(
        pred_prob_tp,
        bins=100,
        histtype="step",
        color="red",
        label=f"Signals (gap={boundary_diff})",
    )

    plt.hist(
        pred_prob_tn,
        bins=100,
        histtype="step",
        color="blue",
        label="Noise",
    )

    plt.yscale("log")
    plt.xlabel("Prediction Probability (Sigmoid)")
    plt.ylabel("Number of Occurrences")
    plt.title(f"Pred Prob Output at {_etitle(epoch)}")
    plt.legend()
    plt.grid(True, ls=":")

    if save and save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, f"log_pred_prob_output_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
