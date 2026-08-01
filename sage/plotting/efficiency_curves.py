#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : efficiency_curves.py
Description     : Short description of the file

Created on 2026-03-21 17:31:11

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


def plot_efficiency_curves(
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
    Plot detection efficiency as a function of each source parameter.

    For each continuous source parameter (chirp mass, distance, SNR, etc.)
    bins the detected fraction (``pred_stat > threshold``) as a function of
    the parameter value and overlays curves for a sweep of thresholds.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier used in the output directory name.
    source_params : dict[str, array-like]
        Dictionary mapping parameter name to per-signal values.
    pred_stat : array-like, shape ``(N_signal,)``
        Predicted ranking statistic for signal events only.
    labels : array-like, shape ``(N,)``
        Binary labels (1 = signal) for the full validation set.
    export_dir : str or None
        Parent directory; plots are saved under
        ``EFFICIENCY/epoch_{epoch}/``.
    save : bool
        If ``True``, save figures to disk; otherwise display interactively.
    save_name : str
        Prefix for saved filenames (default ``"stat"``).
    bin_width : int
        Number of samples per parameter bin (default ``500``).
    step : float
        Step size for threshold sweep (default ``10``).
    """

    if save and export_dir is not None:
        base_dir = os.path.join(export_dir, f"EFFICIENCY/epoch_{epoch}")
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
            continue  # skip pathological epochs

        # --------------------------------------------
        # sort by parameter
        # --------------------------------------------
        order = np.argsort(source_data)
        p_sorted = source_data[order]
        s_sorted = data_tp[order]

        # --------------------------------------------
        # sliding window efficiency proxy
        # --------------------------------------------
        xvals = []
        yvals = []

        for start in range(0, len(p_sorted) - bin_width, step):

            end = start + bin_width

            window_param = p_sorted[start:end]
            window_stat = s_sorted[start:end]

            xvals.append(np.median(window_param))
            yvals.append(np.median(window_stat))

        xvals = np.array(xvals)
        yvals = np.array(yvals)

        # --------------------------------------------
        # plot
        # --------------------------------------------
        plt.figure(figsize=(7, 6))

        plt.plot(
            xvals,
            yvals,
            color="black",
            label=key,
        )

        plt.xlabel(key)
        plt.ylabel(save_name)
        plt.title(f"Efficiency Curve for {key} at {epoch}")
        plt.grid(True, ls=":")
        plt.legend()

        if save and base_dir is not None:
            plt.savefig(
                os.path.join(
                    base_dir,
                    f"efficiency_{save_name}_{key}_{epoch}.png",
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()
            plt.close()
