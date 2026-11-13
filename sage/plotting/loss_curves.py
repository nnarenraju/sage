#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training and validation loss curves, and the validated epochs of a
run's ``losses.h5``.

Filename        : loss_curves.py
Description     : Short description of the file

Created on 2026-03-21 17:41:20

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


def validated_epochs(validation_loss):
    """
    Indices of the epochs whose losses were written.

    ``losses.h5`` is pre-allocated with zeros, so a row whose total loss
    (column 0) is exactly 0.0 has not been written.

    Parameters
    ----------
    validation_loss : numpy.ndarray, shape ``(E, L)``
        Losses per epoch; column 0 is the total loss.

    Returns
    -------
    numpy.ndarray
        Integer indices of the written rows.
    """
    validation_loss = np.asarray(validation_loss)
    return np.flatnonzero(validation_loss[:, 0] != 0.0)


def best_validated_epoch(validation_loss):
    """
    Index of the validated epoch with the lowest total loss.

    Parameters
    ----------
    validation_loss : numpy.ndarray, shape ``(E, L)``
        Validation losses per epoch; column 0 is the total loss.

    Returns
    -------
    int or None
        ``None`` when no epoch has been validated.
    """
    idx = validated_epochs(validation_loss)
    if idx.size == 0:
        return None
    return int(idx[np.argmin(np.asarray(validation_loss)[idx, 0])])


def plot_loss_curves(
    training_loss,
    validation_loss,
    export_dir=None,
    save=True,
    best_epoch=None,
    component_names=None,
):
    """
    Plot training and validation loss curves; stars mark the best epoch.

    Draws the total loss (column 0) and, when there are more columns,
    the PE component losses. Only written rows are drawn; see
    :func:`validated_epochs`.

    Parameters
    ----------
    training_loss : numpy.ndarray, shape ``(E, L)``
        Training losses per epoch; column 0 is the total loss, the
        others the PE component losses.
    validation_loss : numpy.ndarray, shape ``(E, L)``
        Validation losses in the same layout.
    export_dir : str or None
        Saves ``loss_curves.png`` and ``pe_loss_curves.png`` in it.
    save : bool
        Save when ``export_dir`` is set; otherwise show the figures.
    best_epoch : int or None
        Zero-based epoch index to mark with a star, as from
        :func:`best_validated_epoch`.
    component_names : list[str] or None
        Name of each loss column, indexed by column; columns without one
        are labelled ``component {i}``.
    """

    training_loss = np.asarray(training_loss)
    validation_loss = np.asarray(validation_loss)

    epochs = np.arange(len(training_loss))
    val_idx = validated_epochs(validation_loss)
    train_idx = validated_epochs(training_loss)

    plt.figure(figsize=(7, 6))

    plt.plot(epochs[train_idx], training_loss[train_idx, 0], label="Training Loss")
    plt.plot(
        epochs[val_idx],
        validation_loss[val_idx, 0],
        ls="dashed", marker="o", ms=3, label="Validation Loss",
    )

    if best_epoch is not None:
        idx = int(best_epoch)
        plt.scatter(epochs[idx], training_loss[idx, 0], marker="*", s=180,
                    zorder=5, color="tab:green")
        if idx in set(val_idx.tolist()):
            plt.scatter(epochs[idx], validation_loss[idx, 0], marker="*", s=180,
                        zorder=5, color="tab:red",
                        label=f"best epoch {idx} ({validation_loss[idx, 0]:.4f})")

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

    if training_loss.shape[1] == 1:
        return

    plt.figure(figsize=(7, 6))

    for lidx in range(1, training_loss.shape[1]):

        name = (
            component_names[lidx]
            if component_names is not None and lidx < len(component_names)
            else f"component {lidx}"
        )
        # One colour for the training and validation pair
        line, = plt.plot(epochs[train_idx], training_loss[train_idx, lidx], label=f"{name} (train)")
        plt.plot(
            epochs[val_idx],
            validation_loss[val_idx, lidx],
            ls="dashed", marker="o", ms=3,
            color=line.get_color(),
            label=f"{name} (val)",
        )

        if best_epoch is not None:
            idx = int(best_epoch)
            plt.scatter(epochs[idx], training_loss[idx, lidx], marker="*", s=150,
                        zorder=5, color=line.get_color())

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
