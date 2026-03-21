#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : diagonal_compare.py
Description     : Short description of the file

Created on 2026-03-21 17:36:13

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
import matplotlib.cm as cm


def plot_diagonal_compare(
    epoch,
    pred_params,
    true_params,
    network_snrs,
    labels,
    export_dir=None,
    save=True,
):

    if save and export_dir is not None:
        base_dir = os.path.join(export_dir, f"DIAGONAL/epoch_{epoch}")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = None

    # --------------------------------------------
    # signals only
    # --------------------------------------------
    signal_mask = labels == 1.0

    snr_sig = network_snrs[signal_mask]

    cmap = cm.get_cmap("RdYlBu_r")

    for param in pred_params.keys():

        if param == "gw":
            continue

        pred = pred_params[param][signal_mask]
        true = true_params[param][signal_mask]

        # --------------------------------------------
        # order by SNR (nice visual layering)
        # --------------------------------------------
        order = np.argsort(snr_sig)

        pred = pred[order]
        true = true[order]
        snr = snr_sig[order]

        # SNR > 8 subset
        high = snr > 8.0

        # --------------------------------------------
        # FULL scatter
        # --------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 6))

        sc = ax.scatter(
            pred,
            true,
            c=snr,
            cmap=cmap,
            s=20,
            alpha=0.8,
        )

        cbar = fig.colorbar(sc)
        cbar.set_label("Network SNR")

        ax.set_xlabel(f"Predicted [{param}]")
        ax.set_ylabel(f"True [{param}]")
        ax.set_title(f"Diagonal Plot of {param} at {epoch}")
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
        if np.sum(high) == 0:
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
        ax2.set_title(f"Diagonal Plot of {param} (SNR>8) at {epoch}")
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
