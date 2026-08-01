#!/usr/bin/env python3
"""
Corner plot: IMRPhenomXAS_NRTidalv3 mismatch vs LALSim across N random BNS draws.

Parameters shown:  m1, m2, chi1z, chi2z, Lambda1, Lambda2
Colour:            log10(mismatch)

Usage
-----
Run from the sage root directory::

    python -m sage.diagnostics.plot_nrt_corner_mismatch

or::

    python sage/diagnostics/plot_nrt_corner_mismatch.py [--n_samples N] [--seed S] [--out PATH]

The BNS production config (T=295 s, 2048 Hz, f_low=20 Hz) is set up inline;
no external run-config files are required.
"""

import argparse
import math
import os

import numpy as np
import torch

# ---------------------------------------------------------------------------
# BNS production config — inline, no file loading required
# ---------------------------------------------------------------------------
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.core.config import register_configs, get_cfg


def _ensure_bns_config():
    try:
        get_cfg()
    except RuntimeError:
        class _Cfg:
            batch_size    = 2
            device        = "cpu"
            dtype         = torch.float64
            autocast      = False
            class_balance = 0.5

        class _DataCfg:
            sample_rate                  = 2048.0
            signal_low_frequency_cutoff  = 20.0
            noise_low_frequency_cutoff   = 15.0
            sample_length_in_s           = 287.0
            padding_length_in_s          = 4.0

        register_configs(BaseConfig(_Cfg()), BaseDataConfig(_DataCfg()))


_ensure_bns_config()

from sage.data.waveform import IMRPhenomXAS_NRTidalv3   # noqa: E402 (after config)

BNS_SR = 2048.0
BNS_T  = 295.0
BNS_FL = 20.0
BNS_DF = 1.0 / BNS_T
BNS_N  = int(BNS_SR * BNS_T) // 2 + 1   # 302081

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Mismatch computation
# ---------------------------------------------------------------------------

def _lal_hp(m1, m2, chi1z, chi2z, L1, L2, dist, inc, phic):
    from pycbc.waveform import get_fd_waveform
    hp, _ = get_fd_waveform(
        approximant="IMRPhenomXAS_NRTidalv3",
        mass1=float(m1), mass2=float(m2),
        spin1z=float(chi1z), spin2z=float(chi2z),
        lambda1=float(L1), lambda2=float(L2),
        distance=float(dist), delta_f=BNS_DF,
        f_lower=BNS_FL, f_ref=BNS_FL,
        inclination=float(inc), coa_phase=float(phic),
    )
    d = np.array(hp.data, dtype=np.complex128)
    if len(d) < BNS_N:
        d = np.pad(d, (0, BNS_N - len(d)))
    return d[:BNS_N]


def _mismatch_one(model, m1, m2, chi1z, chi2z, L1, L2, dist, inc, phic):
    import pycbc.types
    from pycbc.filter import match as pycbc_match

    th = torch.tensor([[m1, m2, chi1z, chi2z, dist, 0., phic, inc, L1, L2]], dtype=DTYPE)
    with torch.no_grad():
        hp_s, _ = model.get_hphc(th, reproduce_lal=True)
    hp_np = hp_s[0].to(torch.complex128).numpy()

    hp_lal = _lal_hp(m1, m2, chi1z, chi2z, L1, L2, dist, inc, phic)

    psd = pycbc.types.FrequencySeries(np.ones(BNS_N, dtype=np.float64), delta_f=BNS_DF)
    m_val, _ = pycbc_match(
        pycbc.types.FrequencySeries(hp_np,   delta_f=BNS_DF),
        pycbc.types.FrequencySeries(hp_lal,  delta_f=BNS_DF),
        psd=psd, low_frequency_cutoff=BNS_FL,
    )
    return 1.0 - m_val


def compute_mismatches(n_samples=1000, seed=20260604):
    """Draw n_samples random BNS systems and return (params_dict, mismatches)."""
    rng = np.random.default_rng(seed=seed)

    _m = rng.uniform(1, 3, (n_samples, 2)); _m.sort(1); _m = _m[:, ::-1]
    m1   = _m[:, 0].copy()
    m2   = _m[:, 1].copy()
    c1z  = rng.uniform(-0.4, 0.4, n_samples)
    c2z  = rng.uniform(-0.4, 0.4, n_samples)
    lam1 = rng.uniform(0, 5000, n_samples)
    lam2 = rng.uniform(0, 5000, n_samples)
    inc  = np.arccos(rng.uniform(-1, 1, n_samples))
    phic = rng.uniform(0, 2 * math.pi, n_samples)
    dist = rng.uniform(10, 500, n_samples)

    model = IMRPhenomXAS_NRTidalv3()
    mismatches = np.zeros(n_samples)

    print(f"Computing mismatches for {n_samples} BNS samples ...")
    for i in range(n_samples):
        try:
            mismatches[i] = _mismatch_one(
                model, m1[i], m2[i], c1z[i], c2z[i],
                lam1[i], lam2[i], dist[i], inc[i], phic[i],
            )
        except Exception as e:
            mismatches[i] = np.nan
            print(f"  [WARN] sample {i}: {e}")
        if (i + 1) % 100 == 0:
            valid = mismatches[: i + 1][~np.isnan(mismatches[: i + 1])]
            print(
                f"  {i+1}/{n_samples}  "
                f"worst={valid.max():.2e}  median={np.median(valid):.2e}"
            )

    params = dict(m1=m1, m2=m2, c1z=c1z, c2z=c2z, lam1=lam1, lam2=lam2)
    return params, mismatches


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

def make_corner_plot(params, mismatches, out_path):
    """Save a corner plot of log10(mismatch) vs BNS parameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    valid = ~np.isnan(mismatches)
    mm    = mismatches[valid]
    print(
        f"\n  {valid.sum()} valid samples\n"
        f"  mismatch: min={mm.min():.2e}  median={np.median(mm):.2e}  "
        f"max={mm.max():.2e}  >1e-6: {(mm > 1e-6).sum()}"
    )

    PARAMS = [
        (params["m1"][valid],   r"$m_1\ [M_\odot]$",   (1, 3)),
        (params["m2"][valid],   r"$m_2\ [M_\odot]$",   (1, 3)),
        (params["c1z"][valid],  r"$\chi_{1z}$",         (-0.4, 0.4)),
        (params["c2z"][valid],  r"$\chi_{2z}$",         (-0.4, 0.4)),
        (params["lam1"][valid], r"$\Lambda_1$",          (0, 5000)),
        (params["lam2"][valid], r"$\Lambda_2$",          (0, 5000)),
    ]
    N_PAR = len(PARAMS)
    log_mm = np.log10(np.clip(mm, 1e-12, 1.0))
    vmin, vmax = log_mm.min(), log_mm.max()
    cmap = plt.cm.plasma_r

    fig, axes = plt.subplots(N_PAR, N_PAR, figsize=(14, 14))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for row in range(N_PAR):
        for col in range(N_PAR):
            ax = axes[row, col]
            if row < col:
                ax.set_visible(False)
                continue

            x_vals, x_lbl, x_lim = PARAMS[col]
            y_vals, y_lbl, y_lim = PARAMS[row]

            if row == col:
                ax.hist(x_vals, bins=30, color="steelblue", alpha=0.7, density=True)
                ax.set_xlim(x_lim)
            else:
                ax.scatter(x_vals, y_vals, c=log_mm,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           s=3, alpha=0.6, linewidths=0)
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)

            ax.set_xlabel(x_lbl, fontsize=9) if row == N_PAR - 1 else ax.set_xticklabels([])
            ax.set_ylabel(y_lbl, fontsize=9) if col == 0 else ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.add_axes([0.92, 0.15, 0.02, 0.70]))
    cbar.set_label(r"$\log_{10}(\mathrm{mismatch})$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"IMRPhenomXAS_NRTidalv3 vs LALSim  —  {valid.sum()} BNS draws\n"
        f"worst={mm.max():.1e}  median={np.median(mm):.1e}  "
        f"(flat PSD, reproduce_lal=True)",
        fontsize=11, y=0.98,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Number of BNS draws (default: 1000)")
    p.add_argument("--seed", type=int, default=20260604,
                   help="RNG seed (default: 20260604, matches corner_mismatch_nrt.py)")
    p.add_argument("--out", default="diagnostics/corner_mismatch_nrt.png",
                   help="Output PNG path (default: diagnostics/corner_mismatch_nrt.png)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    params, mismatches = compute_mismatches(n_samples=args.n_samples, seed=args.seed)
    make_corner_plot(params, mismatches, out_path=args.out)
