#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P-P calibration plot of the Gaussian point estimates ``N(mu, sigma)``.

For calibrated predictions the probability integral transform (PIT)
``Phi((y - mu) / sigma)`` of the true value ``y`` is uniform on
``[0, 1]``, so its empirical CDF lies on the diagonal.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import ndtr  # standard-normal CDF, Phi
from sage.plotting._epochs import epoch_tag as _etag, epoch_title as _etitle


def _pit(mu, sigma, y, eps=1e-12):
    """
    Probability integral transform ``Phi((y - mu) / (|sigma| + eps))``.

    Parameters
    ----------
    mu, sigma, y : numpy.ndarray
        Predicted means, predicted standard deviations, true values.
    eps : float
        Added to ``|sigma|`` to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        PIT values in ``[0, 1]``.
    """
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
    """
    Plot the empirical CDF of the PIT of each parameter against the
    diagonal.

    Rows with a non-finite value are dropped per parameter.

    Parameters
    ----------
    mu, sigma, y : array-like, shape ``(N,)`` or ``(N, P)``
        Predicted means, predicted standard deviations (not
        log-variances) and true values, for signals only.
    param_names : list[str] or None
        Legend names of the ``P`` parameters; ``param {i}`` when None.
    epoch : int or str or None
        Epoch identifier for the title and filename.
    export_dir : str or None
        Saves ``calibration/pp_calibration_epoch_NNNN.png`` under it.
    save : bool
        Save when ``export_dir`` is set (close unshown when it is not);
        otherwise show the figure.
    title : str or None
        Plot title; None uses the default title.

    Returns
    -------
    dict
        ``{name: {"ks": ks, "cov1sigma": c1, "cov2sigma": c2}}``: the
        Kolmogorov-Smirnov distance of the PIT from uniform, and the
        fractions with ``|y - mu| / sigma`` below 1 and 2.
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
            title += f" — {_etitle(epoch)}"
    plt.title(title)

    if save and export_dir is not None:
        outdir = os.path.join(export_dir, "calibration")
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(
            os.path.join(outdir, f"pp_calibration_{_etag(epoch)}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
    elif save:
        plt.close()
    else:
        plt.show()
        plt.close()

    return metrics
