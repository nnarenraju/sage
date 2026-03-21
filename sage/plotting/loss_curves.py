#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : loss_curves.py
Description     : Short description of the file

Created on 2026-03-21 17:41:20

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


def plot_loss_curves(
    training_loss,
    validation_loss,
    export_dir=None,
    save=True,
    best_epoch=None,
    pe_training_losses=None,
    pe_validation_losses=None,
):

    epochs = np.arange(1, len(training_loss) + 1)

    # --------------------------------------------
    # TOTAL LOSS CURVE
    # --------------------------------------------
    plt.figure(figsize=(7, 6))

    plt.plot(epochs, training_loss, label="Training Loss")
    plt.plot(epochs, validation_loss, ls="dashed", label="Validation Loss")

    if best_epoch is not None:
        idx = int(best_epoch)
        plt.scatter(epochs[idx], training_loss[idx], marker="*", s=150)
        plt.scatter(epochs[idx], validation_loss[idx], marker="*", s=150)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True, ls=":")
    plt.legend()

    if save and export_dir is not None:
        plt.savefig(
            os.path.join(export_dir, "loss_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()

    # --------------------------------------------
    # PARAMETER ESTIMATION LOSS CURVES (optional)
    # --------------------------------------------
    if pe_training_losses is None or pe_validation_losses is None:
        return

    plt.figure(figsize=(7, 6))

    for key in pe_training_losses.keys():

        plt.plot(
            epochs,
            pe_training_losses[key],
            label=f"{key} train",
        )

        plt.plot(
            epochs,
            pe_validation_losses[key],
            ls="dashed",
            label=f"{key} valid",
        )

        if best_epoch is not None:
            idx = int(best_epoch)
            plt.scatter(epochs[idx], pe_training_losses[key][idx], marker="*", s=150)
            plt.scatter(epochs[idx], pe_validation_losses[key][idx], marker="*", s=150)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Parameter Estimation Loss Curves")
    plt.grid(True, ls=":")
    plt.legend()

    if save and export_dir is not None:
        plt.savefig(
            os.path.join(export_dir, "pe_loss_curves.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()
        plt.close()
