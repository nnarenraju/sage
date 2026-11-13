#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cumulative fraction of the signals' distance-cubed volume against
the ranking statistic.

Filename        : cumulative_volume.py
Description     : Short description of the file

Created on 2026-03-21 17:49:02

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


def plot_cumulative_volume(
    epoch,
    ranking_stat,
    labels,
    source_params,
    distance_param="distance",
    export_dir=None,
    save=True,
    bins=50,
):
    """
    Plot the cumulative volume fraction against the ranking statistic.

    Signals are sorted by ranking statistic, highest first; each adds
    ``distance**3``, and the sum is normalised to 1.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier for the title and filename.
    ranking_stat : array-like, shape ``(N,)``
        Network ranking statistics.
    labels : array-like, shape ``(N,)``
        Class labels; signals are ``1``.
    source_params : dict[str, array-like]
        Per-event parameter arrays; must contain ``distance_param``.
    distance_param : str
        Key of the distance array.
    export_dir : str or None
        Saves ``CUMULATIVE_VOLUME/cumulative_volume_epoch_NNNN.png``
        under it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figure.
    bins : int
        Not used.
    """
    signal_mask = labels == 1.0
    output_sig = ranking_stat[signal_mask]
    distances = source_params[distance_param][signal_mask]

    volumes = distances**3

    sort_idx = np.argsort(output_sig)[::-1]
    output_sorted = output_sig[sort_idx]
    vol_sorted = volumes[sort_idx]

    cum_vol = np.cumsum(vol_sorted)
    cum_vol /= cum_vol[-1]

    plt.figure(figsize=(7, 6))
    plt.plot(output_sorted, cum_vol, c="blue", lw=2)
    plt.xlabel("Network Ranking Statistic")
    plt.ylabel("Cumulative Fraction of Accessible Volume")
    plt.title(f"Cumulative Detection Volume - {_etitle(epoch)}")
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "CUMULATIVE_VOLUME")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"cumulative_volume_{_etag(epoch)}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
