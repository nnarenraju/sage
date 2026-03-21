#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : correllation_matrix.py
Description     : Short description of the file

Created on 2026-03-21 17:52:33

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


def plot_correlation_matrix(
    ranking_stat, source_params, labels, export_dir=None, save=True, epoch=None
):
    import seaborn as sns
    import pandas as pd

    # Only signals
    signal_mask = labels == 1.0
    data = {k: v[signal_mask] for k, v in source_params.items()}
    data["ranking_stat"] = ranking_stat[signal_mask]

    df = pd.DataFrame(data)
    corr = df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix - Epoch {epoch}")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "CORRELATION_MATRIX")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"correlation_matrix_epoch_{epoch}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
