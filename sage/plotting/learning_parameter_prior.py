#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : learning_parameter_prior.py
Description     : Short description of the file

Created on 2026-03-21 17:34:06

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
):
    """
    Overlay learned detection efficiency on the source-parameter prior distribution.

    For each source parameter, shows the prior histogram alongside the
    detection fraction curve — highlighting where the prior is dense versus
    where the network actually achieves high detection efficiency.  Useful
    for diagnosing whether efficiency follows the prior or is biased toward
    high-probability regions.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier used in the output subdirectory name.
    source_params : dict[str, array-like]
        Per-signal parameter arrays.
    pred_stat : array-like, shape ``(N_signal,)``
        Predicted ranking statistic for signal events.
    labels : array-like, shape ``(N,)``
        Binary ground-truth labels for the full validation set.
    export_dir : str or None
        Parent directory; plots saved under ``LEARN_PARAMS/epoch_{epoch}/``.
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    save_name : str
        Prefix for saved filenames (default ``"stat"``).
    bin_width : int
        Number of events per parameter bin (default ``500``).
    step : float
        Step size for threshold sweep (default ``10``).
    """

    if save and export_dir is not None:
        base_dir = os.path.join(export_dir, f"LEARN_PARAMS/epoch_{epoch}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    # --------------------------------------------
    # signals only
    # --------------------------------------------
    signal_mask = labels == 1.0
    data_tp = pred_stat[signal_mask]

    for key, param_array in source_params.items():

        source_data = param_array[signal_mask]

        if len(source_data) < bin_width:
            continue

        # --------------------------------------------
        # sort by parameter
        # --------------------------------------------
        order = np.argsort(source_data)
        p_sorted = source_data[order]
        s_sorted = data_tp[order]

        xvals = []
        mean_vals = []
        p05 = []
        p50 = []
        p95 = []

        # --------------------------------------------
        # sliding window statistics
        # --------------------------------------------
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

        # --------------------------------------------
        # plot
        # --------------------------------------------
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
        plt.title(f"Learning {key} at {epoch}")
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
