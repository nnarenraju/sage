#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : diagonal_compare.py
Description     : Short description of the file

Created on 2026-03-21 17:36:13

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
import matplotlib.cm as cm
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle


def plot_diagonal_compare(
    epoch,
    pred_params,
    true_params,
    network_snrs,
    labels,
    export_dir=None,
    save=True,
):
    """
    True-vs-predicted scatter plots for each estimated parameter, coloured by SNR.

    Produces two scatter plots per parameter: the full signal set and a
    high-SNR (> 8) subset.  Points are coloured by network optimal SNR.
    A dashed diagonal reference line marks perfect recovery.

    Parameters
    ----------
    epoch : int or str
        Epoch identifier used in the output subdirectory name.
    pred_params : dict[str, array-like]
        Network point estimates per parameter.
    true_params : dict[str, array-like]
        Ground-truth parameter values in the same layout.
    network_snrs : array-like, shape ``(N,)``
        Optimal network SNR for all validation events.
    labels : array-like, shape ``(N,)``
        Binary labels; signals selected with ``label == 1``.
    export_dir : str or None
        Parent directory; plots saved under ``DIAGONAL/epoch_{epoch}/``.
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    """

    if save and export_dir is not None:
        base_dir = os.path.join(export_dir, f"DIAGONAL/{_etag(epoch)}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    # --------------------------------------------
    # signals only
    # --------------------------------------------
    signal_mask = labels == 1.0

    # Colour-by-SNR is optional: the optimal SNR is computed during signal
    # generation but is not currently recorded in the validation output, so on
    # most runs there is nothing to colour by. Fall back to uncoloured points
    # rather than substituting a different quantity (chirp distance was
    # previously standing in, on an axis labelled SNR).
    has_snr = network_snrs is not None
    snr_sig = np.asarray(network_snrs)[signal_mask] if has_snr else None

    cmap = cm.get_cmap("RdYlBu_r")

    for param in pred_params.keys():

        if param == "gw":
            continue

        pred = pred_params[param][signal_mask]
        true = true_params[param][signal_mask]

        # --------------------------------------------
        # order by SNR (nice visual layering)
        # --------------------------------------------
        if has_snr:
            order = np.argsort(snr_sig)
            snr = snr_sig[order]
            high = snr > 8.0
        else:
            order = np.arange(len(pred))
            snr, high = None, None

        pred = pred[order]
        true = true[order]

        # --------------------------------------------
        # FULL scatter
        # --------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 6))

        if has_snr:
            sc = ax.scatter(pred, true, c=snr, cmap=cmap, s=20, alpha=0.8)
            cbar = fig.colorbar(sc)
            cbar.set_label("Network SNR")
        else:
            # With ~100k signals and nothing to colour by, a scatter saturates
            # into a solid block and hides all density structure. A hexbin
            # shows where the samples actually are.
            hb = ax.hexbin(pred, true, gridsize=60, bins="log", mincnt=1,
                           cmap="viridis")
            cbar = fig.colorbar(hb)
            cbar.set_label("count (log)")

        ax.set_xlabel(f"Predicted [{param}]")
        ax.set_ylabel(f"True [{param}]")
        ax.set_title(f"Diagonal Plot of {param} at {_etitle(epoch)}")
        ax.grid(True, ls=":")

        # diagonal reference
        if param not in ("norm_dist", "norm_dchirp"):
            mn = min(pred.min(), true.min())
            mx = max(pred.max(), true.max())
            ax.plot([mn, mx], [mn, mx], ls="dashed", color="k", label="Ideal")
            ax.legend()

        if save and base_dir is not None:
            fig.savefig(
                os.path.join(base_dir, f"diagonal_{param}_{epoch}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

        # --------------------------------------------
        # HIGH SNR scatter
        # --------------------------------------------
        # The high-SNR panel needs a real SNR; skip it when none was recorded.
        if not has_snr or np.sum(high) == 0:
            continue

        fig2, ax2 = plt.subplots(figsize=(7, 6))

        sc2 = ax2.scatter(
            pred[high],
            true[high],
            c=snr[high],
            cmap=cmap,
            s=20,
            alpha=0.8,
        )

        cbar2 = fig2.colorbar(sc2)
        cbar2.set_label("Network SNR")

        ax2.set_xlabel(f"Predicted [{param}]")
        ax2.set_ylabel(f"True [{param}]")
        ax2.set_title(f"Diagonal Plot of {param} (SNR>8) at {_etitle(epoch)}")
        ax2.grid(True, ls=":")

        if param not in ("norm_dist", "norm_dchirp"):
            mn = min(pred[high].min(), true[high].min())
            mx = max(pred[high].max(), true[high].max())
            ax2.plot([mn, mx], [mn, mx], ls="dashed", color="k", label="Ideal")
            ax2.legend()

        if save and base_dir is not None:
            fig2.savefig(
                os.path.join(base_dir, f"diagonal_snr_gt8_{param}_{epoch}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig2)
        else:
            plt.show()
            plt.close(fig2)
