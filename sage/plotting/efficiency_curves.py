#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Running median of the ranking statistic of the signals against each
source parameter.

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

import os
import numpy as np
import matplotlib.pyplot as plt
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle
from sage.plotting._params import select_params as _select_params


def plot_efficiency_curves(
    epoch,
    source_params,
    pred_stat,
    labels,
    export_dir=None,
    save=True,
    save_name="stat",
    bin_width=None,
    step=None,
    params=None,
):
    """
    Plot the running median of the signals' ranking statistic against
    each source parameter.

    Signals are sorted by the parameter; each sliding window gives one
    point (median parameter, median statistic). A parameter with fewer
    signals than one window is skipped.

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
        Saves the plots under ``EFFICIENCY/epoch_NNNN/`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figures.
    save_name : str
        y-axis label and file name tag.
    bin_width : int or None
        Signals per window; ``None`` uses ``max(500, N_signal // 50)``.
    step : int or None
        Stride between windows; ``None`` uses ``bin_width // 4``.
    params : iterable[str] or None
        Source parameters to plot; see
        :func:`~sage.plotting._params.select_params`.
    """

    if save and export_dir is not None:
        base_dir = os.path.join(export_dir, f"EFFICIENCY/{_etag(epoch)}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    signal_mask = labels == 1.0
    data_tp = pred_stat[signal_mask]

    n_sig = int(signal_mask.sum())
    if bin_width is None:
        bin_width = max(500, n_sig // 50)
    if step is None:
        step = max(1, bin_width // 4)

    for key, param_array in _select_params(source_params, params).items():

        source_data = param_array[signal_mask]

        if len(source_data) < bin_width:
            continue

        order = np.argsort(source_data)
        p_sorted = source_data[order]
        s_sorted = data_tp[order]

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

        plt.figure(figsize=(7, 6))

        plt.plot(
            xvals,
            yvals,
            color="black",
            label=key,
        )

        plt.xlabel(key)
        plt.ylabel(save_name)
        plt.title(f"Efficiency Curve for {key} at {_etitle(epoch)}")
        plt.grid(True, ls=":")
        plt.legend()

        if save and base_dir is not None:
            plt.savefig(
                os.path.join(
                    base_dir,
                    f"efficiency_{save_name}_{key}_{_etag(epoch)}.png",
                ),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()
            plt.close()
