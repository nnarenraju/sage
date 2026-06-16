#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consistency-model training diagnostics: a single figure that puts the per-term
loss curves and the per-term gradient-norm balance side by side.

The multi-detector consistency model trains several objectives at once — BCE
classification, the coherent point-estimate NLL, the coupling term, and the
per-detector tc/mchirp consistency NLLs. Two things must be watched together:

  * the *loss* curves (is each term learning, is anything spiking?), and
  * the *gradient-norm* contribution of each term relative to BCE (is any
    auxiliary term out-gradienting the classifier on the shared parameters?).

:func:`plot_consistency_diagnostics` renders both in one figure, plus the scale
of the corroboration features entering the ranking head.

All inputs are plain arrays (per-iteration histories) so the routine is decoupled
from how they are collected — the training loop logs them, or an offline audit
produces them.
"""

import numpy as np


_MERGED_COLS = ("total", "bce", "pe_reg", "coupling")
_CONS_COLS = ("total", "tc", "mc")
_FEAT_COLS = ("backbone", "corr_mean", "corr_max")


def _roll(a, w):
    a = np.asarray(a, dtype=float)
    if w <= 1 or len(a) < w:
        return np.arange(len(a)), a
    k = np.convolve(a, np.ones(w) / w, mode="valid")
    return np.arange(len(a) - len(k), len(a)), k


def _curve(ax, y, label, smooth):
    ax.plot(np.arange(len(y)), y, alpha=0.2)
    x, k = _roll(y, smooth)
    ax.plot(x, k, label=label)


def plot_consistency_diagnostics(
    merged,
    cons,
    feats=None,
    grad_xbce=None,
    grad_target=0.33,
    cons_weight=0.1,
    epoch_boundaries=(),
    smooth=25,
    title=None,
    save_path=None,
):
    """Combined loss + gradient-norm diagnostics for consistency training.

    Parameters
    ----------
    merged : array, shape ``(N, 4)``
        Per-iteration merged-loss components ``[total, bce, pe_reg, coupling]``
        (the raw, pre-weight values returned by ``BCEWithPEsigmaLoss``).
    cons : array, shape ``(N, 3)``
        Per-iteration consistency-loss components ``[total, tc, mc]``.
    feats : array, shape ``(N, 3)``, optional
        Per-iteration ranking-head input scales ``[backbone_meanabs,
        corr_meanabs, corr_maxabs]``.
    grad_xbce : dict {term: value}, optional
        Per-term gradient norm as a multiple of the BCE gradient norm
        (e.g. ``{"cons_mc": 0.12, "cons_tc": 0.04, ...}``). Rendered as the
        gradient-balance bar panel with the ``grad_target`` ceiling line.
    grad_target : float
        The per-/total-aux gradient-norm ceiling (fraction of BCE) to mark.
    epoch_boundaries : sequence of int
        Iteration indices to mark with vertical lines (epoch transitions).
    smooth : int
        Rolling-mean window for the noisy per-iteration curves.
    title : str, optional
    save_path : str or Path, optional
        If given, the figure is saved here (dpi 110).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merged = np.asarray(merged, dtype=float)
    cons = np.asarray(cons, dtype=float)
    N = len(merged)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))

    def _epochs(a):
        for b in epoch_boundaries:
            a.axvline(b, ls="--", c="k", alpha=0.35)

    # (0,0) classification / coherent-PE loss components
    a = ax[0, 0]
    _curve(a, merged[:, 1], "BCE", smooth)
    _curve(a, merged[:, 2], "PE_reg", smooth)
    _curve(a, merged[:, 3], "coupling", smooth)
    _epochs(a); a.set_title("classification / coherent-PE losses")
    a.set_xlabel("iter"); a.legend(); a.grid(alpha=0.3)

    # (0,1) per-detector consistency NLL
    a = ax[0, 1]
    _curve(a, cons[:, 1], "cons_tc", smooth)
    _curve(a, cons[:, 2], "cons_mc", smooth)
    _epochs(a); a.set_title("per-detector consistency NLL")
    a.set_xlabel("iter"); a.legend(); a.grid(alpha=0.3)

    # (0,2) total losses
    a = ax[0, 2]
    _curve(a, merged[:, 0], "merged total", smooth)
    _curve(a, cons[:, 0], "consistency total", smooth)
    _epochs(a); a.set_title("total losses")
    a.set_xlabel("iter"); a.legend(); a.grid(alpha=0.3)

    # (1,0) gradient-norm balance (per term, as multiple of BCE)
    a = ax[1, 0]
    if grad_xbce:
        terms = ["bce"] + [t for t in grad_xbce if t != "bce"]
        vals = [grad_xbce.get("bce", 1.0)] + [grad_xbce[t] for t in terms[1:]]
        colors = ["tab:blue"] + ["tab:orange" if t.startswith("cons") else "tab:gray"
                                 for t in terms[1:]]
        a.bar(terms, vals, color=colors)
        a.axhline(grad_target, ls="--", c="r", label=f"{grad_target:g}x BCE ceiling")
        nonbce = sum(v for t, v in zip(terms, vals) if t != "bce")
        a.text(0.5, 0.9, f"non-BCE combined <= {nonbce:.2f}x BCE",
               transform=a.transAxes, ha="center",
               bbox=dict(fc="lightyellow", ec="gray"))
        a.set_ylabel("||grad|| / ||grad BCE||"); a.legend()
    else:
        a.text(0.5, 0.5, "no gradient-norm data", ha="center", transform=a.transAxes)
    a.set_title("gradient-norm balance (xBCE)"); a.grid(alpha=0.3, axis="y")

    # (1,1) ranking-head input feature scales
    a = ax[1, 1]
    if feats is not None:
        feats = np.asarray(feats, dtype=float)
        a.plot(feats[:, 0], label="backbone |feat| mean")
        a.plot(feats[:, 1], label="corr |feat| mean")
        a.plot(feats[:, 2], label="corr |feat| max", alpha=0.5)
        a.set_yscale("log"); _epochs(a); a.legend()
    else:
        a.text(0.5, 0.5, "no feature-scale data", ha="center", transform=a.transAxes)
    a.set_title("ranking-head input feature scales"); a.set_xlabel("iter"); a.grid(alpha=0.3)

    # (1,2) weighted contribution to total loss value (final-window mean)
    a = ax[1, 2]
    w = max(1, N // 5)
    W_REG, W_COUP = 0.005, 0.005   # fixed merged-loss weights
    bce = merged[-w:, 1].mean()
    bars = {
        "BCE": bce,
        "PE_reg": W_REG * merged[-w:, 2].mean(),
        "coupling": W_COUP * merged[-w:, 3].mean(),
        "consistency": cons_weight * cons[-w:, 0].mean(),
    }
    a.bar(list(bars), list(bars.values()),
          color=["tab:blue", "tab:gray", "tab:gray", "tab:orange"])
    a.set_title(f"weighted loss-value contribution (last {w} iters)")
    a.set_ylabel("weighted loss value"); a.grid(alpha=0.3, axis="y")

    fig.suptitle(title or "Consistency training diagnostics")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), dpi=110)
    return fig
