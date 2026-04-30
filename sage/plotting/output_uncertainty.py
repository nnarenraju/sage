#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : output_uncertainty.py
Description     : Short description of the file

Created on 2026-03-21 17:51:47

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


def plot_output_vs_uncertainty(
    model,
    source_params,
    labels,
    export_dir=None,
    save=True,
    epoch=None,
):
    """
    Plot network ranking statistic vs model uncertainty.
    """
    signal_mask = labels == 1.0
    input_dict = {k: v for k, v in source_params.items()}

    # Inference
    ranking_stat, uncertainty = model.predict(input_dict, return_uncertainty=True)
    ranking_stat = ranking_stat[signal_mask]
    uncertainty = uncertainty[signal_mask]

    plt.figure(figsize=(7, 6))
    plt.scatter(ranking_stat, uncertainty, alpha=0.5, c="red", label="Signals")
    plt.xlabel("Ranking Statistic")
    plt.ylabel("Predicted Uncertainty")
    plt.title(f"Output vs Uncertainty - Epoch {epoch}")
    plt.grid(True, ls=":")
    plt.legend()

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "OUTPUT_VS_UNCERTAINTY")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"output_vs_uncertainty_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
