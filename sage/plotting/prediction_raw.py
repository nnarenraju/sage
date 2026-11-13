#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Histograms of the ranking statistic per class, and the fraction of
signals above each noise ranking statistic.

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

import os
import numpy as np
import matplotlib.pyplot as plt
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle


def recovered_fraction(raw_tp, raw_tn):
    """
    Fraction of signals strictly above each noise ranking statistic.

    Parameters
    ----------
    raw_tp, raw_tn : numpy.ndarray
        Signal and noise ranking statistics.

    Returns
    -------
    thresholds : numpy.ndarray
        The noise ranking statistics, highest first.
    fraction : numpy.ndarray
        Fraction of signals above each threshold.
    """
    thresholds = np.sort(raw_tn)[::-1]
    above = len(raw_tp) - np.searchsorted(np.sort(raw_tp), thresholds, side="right")
    return thresholds, above / len(raw_tp)


def plot_prediction_raw(epoch, ranking_stat, labels, export_dir=None, save=True):
    """
    Plot ranking-statistic histograms per class and the fraction of
    signals above each noise ranking statistic.

    Left panel: log-count histograms of signals, noise and, when
    present, non-astrophysical events. Right panel:
    :func:`recovered_fraction`, with a secondary axis in signal counts.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Ranking statistic of every event.
    labels : array-like, shape ``(N,)``
        Class labels (0 noise, 1 signal, 2 non-astrophysical).
    export_dir : str or None
        Saves the plot under ``PRED_RAW/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    """

    save_dir = None
    if save and export_dir is not None:
        save_dir = os.path.join(export_dir, "PRED_RAW")
        os.makedirs(save_dir, exist_ok=True)

    signal_mask = labels == 1.0
    noise_mask = labels == 0.0

    raw_tp = ranking_stat[signal_mask]
    raw_tn = ranking_stat[noise_mask]

    frac_detected = np.column_stack(recovered_fraction(raw_tp, raw_tn))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Raw Output at {_etitle(epoch)}")

    # float64: 100 finite bins even for values a float32 ulp apart
    ax[0].hist(np.asarray(raw_tp, dtype=np.float64), bins=100, histtype="step", label="Signals")
    ax[0].hist(np.asarray(raw_tn, dtype=np.float64), bins=100, histtype="step", label="Noise")
    if np.any(labels == 2.0):
        raw_na = np.asarray(ranking_stat[labels == 2.0], dtype=np.float64)
        ax[0].hist(raw_na, bins=100, histtype="step", label="Non-astrophysical")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Ranking Statistic")
    ax[0].set_ylabel("Number of Occurrences")
    ax[0].legend()
    ax[0].grid(True, ls=":")

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

    # Secondary axis in number of signals
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


def plot_prediction_raw_comparison(sources, name, export_dir=None, save=True):
    """
    :func:`plot_prediction_raw` for several sources on the same panels.

    One line style per source, one colour per class. The secondary axis
    is drawn only when all sources have the same number of signals.

    Parameters
    ----------
    sources : dict
        Source label -> ``(ranking_stat, labels)``, labels as in
        :func:`plot_prediction_raw`.
    name : str
        Title and filename tag.
    export_dir : str or None
        Saves ``PRED_RAW/log_raw_output_{name}.png`` under it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.

    Raises
    ------
    ValueError
        If there are more than 4 sources.
    """
    classes = [(1.0, "Signals", "C0"), (0.0, "Noise", "C1"), (2.0, "Non-astrophysical", "C2")]
    styles = ["-", "--", ":", "-."]
    if len(sources) > len(styles):
        raise ValueError(f"at most {len(styles)} sources, got {len(sources)}")
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Raw Output, {name}")
    n_signals = set()
    for style, (source, (ranking_stat, labels)) in zip(styles, sources.items()):
        for value, label, colour in classes:
            if np.any(labels == value):
                ax[0].hist(np.asarray(ranking_stat[labels == value], dtype=np.float64), bins=100,
                           histtype="step", color=colour, ls=style, label=f"{label}, {source}")
        raw_tp = ranking_stat[labels == 1.0]
        thresholds, fraction = recovered_fraction(raw_tp, ranking_stat[labels == 0.0])
        ax[1].plot(thresholds, fraction, color="red", ls=style,
                   label=f"{source}, top FAR bin: {int(fraction[0] * len(raw_tp))}/{len(raw_tp)}")
        n_signals.add(len(raw_tp))
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Ranking Statistic")
    ax[0].set_ylabel("Number of Occurrences")
    ax[0].legend()
    ax[0].grid(True, ls=":")
    ax[1].set_xlabel("Noise Stat Threshold")
    ax[1].set_ylabel("Frac Signals Detected above Threshold")
    ax[1].grid(True, ls=":")
    ax[1].legend()
    if len(n_signals) == 1:
        total = n_signals.pop()
        secay = ax[1].secondary_yaxis("right", functions=(lambda frac: frac * total, lambda num: num / total))
        secay.set_ylabel("Num Signals Detected above Threshold")
    plt.tight_layout()
    if save and export_dir is not None:
        save_dir = os.path.join(export_dir, "PRED_RAW")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"log_raw_output_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()
