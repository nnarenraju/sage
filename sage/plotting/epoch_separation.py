#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : epoch_separation.py
Description     : Short description of the file

Created on 2026-03-21 17:57:49

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


def plot_separation_over_epochs(
    all_network_outputs,  # dict epoch -> outputs
    all_labels,  # dict epoch -> labels
    epochs,  # list of epochs
    export_dir=None,
    save=True,
):
    """
    Track separation of signals vs noise across epochs.
    """
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("viridis", len(epochs))

    for i, epoch in enumerate(epochs):
        output = all_network_outputs[epoch]
        labels = all_labels[epoch]

        sig = output[labels == 1.0]
        noise = output[labels == 0.0]

        # KDEs
        sns.kdeplot(sig, label=f"Signals Epoch {epoch}", color=colors[i], linestyle="-")
        sns.kdeplot(
            noise, label=f"Noise Epoch {epoch}", color=colors[i], linestyle="--"
        )

    plt.xlabel("Network Ranking Statistic")
    plt.ylabel("Density")
    plt.title("Signal vs Noise Separation Over Epochs")
    plt.legend(fontsize=8)
    plt.grid(True, ls=":")

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "SEPARATION_OVER_EPOCHS")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, "signal_noise_separation_over_epochs.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
