#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Running mean and percentiles of the ranking statistic of the signals
against each source parameter.

Filename        : learning_parameter_prior.py
Description     : Short description of the file

Created on 2026-03-21 17:34:06

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
from sage.plotting._params import select_params as _select_params


def plot_learning_parameter_prior(
    epoch,
    source_params,
    pred_stat,
    labels,
    export_dir=None,
    save=True,
    save_name="stat",
    bin_width=500,
    step=10,
    params=None,
):
    """
    Plot sliding-window statistics of the signals' ranking statistic
    against each source parameter.

    Per window of signals sorted by the parameter: the mean, the 5-95
    percentile band and the band between median and mean. A parameter
    with fewer signals than one window is skipped.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the titles and file names.
    source_params : dict[str, array-like]
        Parameter name to the value of every event, shape ``(N,)``.
    pred_stat : array-like, shape ``(N,)``
        Ranking statistic of every event.
    labels : array-like, shape ``(N,)``
        Class labels; signals are ``1``.
    export_dir : str or None
        Saves the plots under ``LEARN_PARAMS/epoch_NNNN/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figures.
    save_name : str
        y-axis label and file name tag.
    bin_width : int
        Signals per window.
    step : int
        Stride between windows.
    params : iterable[str] or None
        Source parameters to plot; see
        :func:`~sage.plotting._params.select_params`.
    """

    if save and export_dir is not None:
        base_dir = os.path.join(export_dir, f"LEARN_PARAMS/{_etag(epoch)}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    signal_mask = labels == 1.0
    data_tp = pred_stat[signal_mask]

    for key, param_array in _select_params(source_params, params).items():

        source_data = param_array[signal_mask]

        if len(source_data) < bin_width:
            continue

        order = np.argsort(source_data)
        p_sorted = source_data[order]
        s_sorted = data_tp[order]

        xvals = []
        mean_vals = []
        p05 = []
        p50 = []
        p95 = []

        for start in range(0, len(p_sorted) - bin_width, step):

            end = start + bin_width

            window_param = p_sorted[start:end]
            window_stat = s_sorted[start:end]

            xvals.append(np.median(window_param))
            mean_vals.append(np.mean(window_stat))
            p05.append(np.percentile(window_stat, 5))
            p50.append(np.percentile(window_stat, 50))
            p95.append(np.percentile(window_stat, 95))

        xvals = np.array(xvals)
        mean_vals = np.array(mean_vals)
        p05 = np.array(p05)
        p50 = np.array(p50)
        p95 = np.array(p95)

        plt.figure(figsize=(7, 6))

        plt.plot(
            xvals,
            mean_vals,
            color="blue",
            label="Mean stat",
        )

        plt.fill_between(
            xvals,
            p05,
            p95,
            color="red",
            alpha=0.25,
            label="5-95 %",
        )

        plt.fill_between(
            xvals,
            p50,
            mean_vals,
            color="green",
            alpha=0.25,
            label="Median band",
        )

        plt.xlabel(key)
        plt.ylabel(save_name)
        plt.title(f"Learning {key} at {_etitle(epoch)}")
        plt.grid(True, ls=":")
        plt.legend()

        if save and base_dir is not None:
            plt.savefig(
                os.path.join(
                    base_dir,
                    f"learning_{save_name}_{key}_{epoch}.png",
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()
            plt.close()
