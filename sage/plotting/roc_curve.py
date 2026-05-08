#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : roc_curve.py
Description     : Short description of the file

Created on 2026-03-21 17:20:08

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
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc_curve(epoch, ranking_stat, labels, export_dir=None, save=True):
    """
    Plot and optionally save a log-log ROC curve for one validation epoch.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier used in the plot title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistic (higher = more signal-like).
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels (1 = signal, 0 = noise).
    export_dir : str or None
        Directory to save the figure.  A ``ROC/`` subdirectory is created
        automatically.  Required when ``save=True``.
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    """

    fpr, tpr, _ = metrics.roc_curve(labels, ranking_stat)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(7, 6))

    plt.plot(fpr, tpr, color="red", label=f"AUC = {roc_auc:.5f}")
    plt.plot([0, 1], [0, 1], ls="dashed", color="blue", label="Random Classifier")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(right=1)
    plt.ylim(top=1)
    plt.title(f"ROC Curve at {epoch}")
    plt.legend()
    plt.grid(True, which="both", ls=":")

    if save and export_dir is not None:
        save_dir = os.path.join(export_dir, "ROC")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f"roc_curve_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
    else:
        plt.show()

    plt.close()
