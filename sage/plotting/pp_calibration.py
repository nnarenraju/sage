#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P-P (probability-integral-transform) calibration plot for the heteroscedastic
parameter heads.

For a well-calibrated Gaussian prediction ``N(mu, sigma)`` the PIT value
``z = Phi((y - mu) / sigma)`` of the true target ``y`` is uniformly distributed
on ``[0, 1]``. Plotting the empirical CDF of the PIT values against the diagonal
therefore reveals miscalibration of the predicted uncertainties:

  - curve on the diagonal           -> calibrated
  - curve shallower than diagonal   -> over-confident (sigma too small)
  - curve steeper than diagonal     -> under-confident (sigma too large)

This validates the sigma mechanism that the multi-detector consistency
statistic relies on (it is uncertainty-weighted, so trustworthy sigmas matter),
and works equally for the merged heteroscedastic heads. It complements
``plot_calibration_curve`` (which calibrates the *classifier*).
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import ndtr  # standard-normal CDF, Phi


def _pit(mu, sigma, y, eps=1e-12):
    """Probability integral transform ``Phi((y - mu) / sigma)``."""
    return ndtr((y - mu) / (np.abs(sigma) + eps))


def plot_pp_calibration(
    mu,
    sigma,
    y,
    param_names=None,
    epoch=None,
    export_dir=None,
    save=True,
    title=None,
):
    """P-P calibration plot of heteroscedastic predictions.

    Parameters
    ----------
    mu, sigma, y : array-like, shape ``(N,)`` or ``(N, P)``
        Predicted means, predicted standard deviations (NOT log-variances), and
        the true targets, for ``N`` samples and optionally ``P`` parameters.
        Pass only the supervised samples (e.g. signals); mask out noise first.
    param_names : list[str] or None
        Names for the ``P`` parameters (used in the legend).
    epoch : int or str or None
        Epoch identifier for the title / filename.
    export_dir : str or None
        If given (and ``save``), writes ``calibration/pp_calibration_{epoch}.png``.
    save : bool
        Save to disk if True, else show interactively.
    title : str or None
        Override the default title.

    Returns
    -------
    dict
        Per-parameter calibration metrics: ``{name: {"ks": float,
        "cov1sigma": float, "cov2sigma": float}}`` where ``ks`` is the
        Kolmogorov-Smirnov distance of the PIT from uniform (0 = perfect).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if mu.ndim == 1:
        mu, sigma, y = mu[:, None], sigma[:, None], y[:, None]
    n, p = mu.shape
    names = list(param_names) if param_names is not None else [f"param {i}" for i in range(p)]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], ls="--", color="k", label="Perfect calibration")

    metrics = {}
    for i in range(p):
        good = np.isfinite(mu[:, i]) & np.isfinite(sigma[:, i]) & np.isfinite(y[:, i])
        z = (y[good, i] - mu[good, i]) / (np.abs(sigma[good, i]) + 1e-12)
        pit = _pit(mu[good, i], sigma[good, i], y[good, i])
        pit_sorted = np.sort(pit)
        ecdf = np.arange(1, pit_sorted.size + 1) / pit_sorted.size
        # KS distance of the PIT empirical CDF from the uniform diagonal
        ks = float(np.max(np.abs(ecdf - pit_sorted))) if pit_sorted.size else float("nan")
        cov1 = float(np.mean(np.abs(z) < 1.0)) if z.size else float("nan")
        cov2 = float(np.mean(np.abs(z) < 2.0)) if z.size else float("nan")
        metrics[names[i]] = {"ks": ks, "cov1sigma": cov1, "cov2sigma": cov2}
        plt.plot(
            pit_sorted, ecdf,
            label=f"{names[i]}  (KS={ks:.3f}, 1σ:{cov1:.0%}, 2σ:{cov2:.0%})",
        )

    plt.xlabel(r"PIT  $\Phi((y-\mu)/\sigma)$")
    plt.ylabel("Empirical CDF")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, ls=":")
    plt.legend(loc="upper left", fontsize=9)
    if title is None:
        title = "σ-calibration P–P plot"
        if epoch is not None:
            title += f" — epoch {epoch}"
    plt.title(title)

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "calibration")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"pp_calibration_epoch_{epoch}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
    elif save:
        plt.close()
    else:
        plt.show()
        plt.close()

    return metrics
