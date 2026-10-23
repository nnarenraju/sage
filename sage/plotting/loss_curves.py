#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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

# Packages
import os
import numpy as np
import matplotlib.pyplot as plt


def validated_epochs(validation_loss):
    """Indices of epochs where validation actually ran.

    ``losses.h5`` is pre-allocated ``(num_epochs, num_components)`` and filled
    in as the run proceeds, but validation typically runs only every Nth epoch.
    The rows in between keep their initial **zero**, which is not a loss value.

    Plotting those zeros draws the validation curve as a comb dropping to the
    axis between validation points, and ``argmin`` over the raw column returns
    the first unwritten row rather than the best epoch. Both were happening.

    A total loss of exactly 0.0 is not attainable for the losses used here, so
    an exact zero is an unambiguous "not written yet" sentinel.

    Parameters
    ----------
    validation_loss : numpy.ndarray, shape ``(E, L)``

    Returns
    -------
    numpy.ndarray
        Integer indices of the epochs that were validated.
    """
    validation_loss = np.asarray(validation_loss)
    return np.flatnonzero(validation_loss[:, 0] != 0.0)


def best_validated_epoch(validation_loss):
    """Index of the lowest-total-loss epoch, ignoring un-validated rows.

    Returns ``None`` when nothing has been validated yet.
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
    Plot training and validation loss curves with optional best-epoch markers.

    Generates two figures when ``training_loss`` has more than one column:

    1. **Total loss** — column 0 of both arrays.
    2. **Per-parameter PE losses** — remaining columns.

    Parameters
    ----------
    training_loss : numpy.ndarray, shape ``(E, L)``
        Training losses per epoch; column 0 is the total loss, subsequent
        columns are individual PE component losses.
    validation_loss : numpy.ndarray, shape ``(E, L)``
        Validation losses in the same layout.
    export_dir : str or None
        Directory to save figures (``loss_curves.png``,
        ``pe_loss_curves.png``).
    save : bool
        If ``True``, save to disk; otherwise display interactively.
    best_epoch : int or None
        Zero-based epoch index to mark with a star on all curves. Use
        :func:`best_validated_epoch`, which ignores un-validated rows.
    component_names : list[str] or None
        Names for the PE loss components (columns 1..L-1). Falls back to
        ``component {i}`` when not supplied.

    Notes
    -----
    Only epochs where validation actually ran are drawn for the validation
    curves -- see :func:`validated_epochs`.
    """

    training_loss = np.asarray(training_loss)
    validation_loss = np.asarray(validation_loss)

    epochs = np.arange(1, len(training_loss) + 1)
    # Validation usually runs every Nth epoch; the rows in between are still
    # zero from pre-allocation and must not be drawn. See validated_epochs.
    val_idx = validated_epochs(validation_loss)

    # --------------------------------------------
    # TOTAL LOSS CURVE
    # --------------------------------------------
    plt.figure(figsize=(7, 6))

    plt.plot(epochs, training_loss[:, 0], label="Training Loss")
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

    # --------------------------------------------
    # PARAMETER ESTIMATION LOSS CURVES (optional)
    # --------------------------------------------
    if training_loss.shape[1] == 1:
        return

    plt.figure(figsize=(7, 6))

    for lidx in range(1, training_loss.shape[1]):

        name = (
            component_names[lidx]
            if component_names is not None and lidx < len(component_names)
            else f"component {lidx}"
        )
        # Share a colour between the train/val pair so they read together.
        line, = plt.plot(epochs, training_loss[:, lidx], label=f"{name} (train)")
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
