#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_fiducial.py -- combined-run, line-notched fiducial ASDs (O3a + O3b).

Builds the production fiducial ASDs shared by BOTH the O3a and O3b runs, from the
on-disk recolour banks (``raw_{det}_psds.bin``) -- NO noise re-sampling, NO 16 GB
bank regeneration. For each detector it pools the O3a and O3b banks and computes,
EXACTLY over every pooled segment (column-blocked, no subsampling):

    * median  -- the broadband floor (median over pooled segments), and
    * p99.9   -- the ROBUST worst-case (not the glitch-dominated true max).

It then applies :class:`~sage.data.asd.blackout.LocalLineNotch` (K=4): only narrow
spectral lines (worst spikes above its own local continuum) are notched -- the
fiducial there is raised to ``K * worst`` so the whitened line is capped at
~``1/K`` even on its loudest segment -- while the broadband floor is left at the
median.

Writes ``fiducial_{H1,L1,V1}_psd.bin`` + ``.json`` to ``FIDUCIAL_DIR`` (absolute,
shared). Run once on a CPU node before training:

    python runs/build_fiducial.py
"""

import os
import json

import numpy as np

from sage.utils.servers import get_server
from sage.data.asd.blackout import LocalLineNotch

# Combined O3a+O3b fiducial location -- MUST match config_base.FIDUCIAL_DIR in
# both runs/o3a and runs/o3b.
FIDUCIAL_DIR = "/work/nagarajan/sage_runs/fiducial_psds_o3ab"

RUNS = ["O3a", "O3b"]
DETECTORS = ["H1", "L1", "V1"]

K_DEPTH = 4.0            # fiducial at a line bin = K_DEPTH * worst  -> whitened line <= ~1/K
PCTL = 99.9             # robust worst-case percentile (NOT the true max)
BLOCK = 2048            # frequency-bin block for the exact column-wise percentile
ROWCHUNK = 50_000       # segments per sequential read when filling a block buffer


def _paths(run, det):
    rec = os.path.join(get_server().data_dir(run), "recolour_psds")
    return os.path.join(rec, f"raw_{det}_psds.bin"), os.path.join(rec, f"raw_{det}_psds.json")


def _open_bank(run, det):
    bin_path, meta_path = _paths(run, det)
    with open(meta_path) as f:
        meta = json.load(f)
    n, F = int(meta["num_psds"]), int(meta["num_freq_bins"])
    expected = n * F * 4
    actual = os.path.getsize(bin_path)
    if actual != expected:
        raise RuntimeError(
            f"{run}/{det}: bank {actual} bytes != expected {expected} "
            f"(n={n}, F={F}); bank may be incomplete."
        )
    mm = np.memmap(bin_path, dtype=np.float32, mode="r", shape=(n, F))
    return mm, n, F, float(meta["delta_f"]), float(meta["sample_rate"])


def build_detector(det):
    # Memory-bounded + fast: banks stay as memmaps (their pages live in the OS page
    # cache, reclaimable). For each frequency block we fill one small buffer of the
    # pooled segments by reading SEQUENTIAL row-chunks (fast; a whole-file strided
    # column read is ~100x slower and holding both 16 GB banks resident OOMs), then
    # take the exact median + p99.9 in one pass. No subsampling.
    banks, F0, df0, sr0 = [], None, None, None
    for run in RUNS:
        mm, n, F, df, sr = _open_bank(run, det)
        if F0 is None:
            F0, df0, sr0 = F, df, sr
        elif (F, df) != (F0, df0):
            raise RuntimeError(
                f"{det}: grid mismatch {run} (F={F}, df={df}) vs (F={F0}, df={df0})"
            )
        banks.append((mm, n))
    ntot = sum(n for _, n in banks)
    print(f"[{det}] pooling " + " + ".join(f"{run}:{n}" for run, (_, n) in zip(RUNS, banks))
          + f" = {ntot} segments (F={F0}); exact median + p{PCTL} ...", flush=True)

    median = np.empty(F0, dtype=np.float64)
    worst = np.empty(F0, dtype=np.float64)
    for b0 in range(0, F0, BLOCK):
        b1 = min(F0, b0 + BLOCK)
        buf = np.empty((ntot, b1 - b0), dtype=np.float32)      # ~4 GB at BLOCK=2048
        off = 0
        for mm, n in banks:
            for r0 in range(0, n, ROWCHUNK):
                r1 = min(n, r0 + ROWCHUNK)
                buf[off + r0:off + r1] = mm[r0:r1, b0:b1]      # sequential region read
            off += n
        # q=50 (linear interp) == np.median for even n; overwrite_input avoids a copy.
        res = np.percentile(buf, [50.0, PCTL], axis=0, overwrite_input=True)
        median[b0:b1] = res[0]
        worst[b0:b1] = res[1]
        del buf
        print(f"   [{det}] bins {b1}/{F0}", flush=True)

    freqs = np.arange(F0, dtype=np.float64) * df0
    policy = LocalLineNotch(freqs, k_depth=K_DEPTH)     # 8 Hz window, thr 2x, taper 1.5, 15-1024 Hz
    fiducial, line_idx = policy.apply(median, worst)
    fiducial = np.asarray(fiducial, dtype=np.float32)

    os.makedirs(FIDUCIAL_DIR, exist_ok=True)
    bin_out = os.path.join(FIDUCIAL_DIR, f"fiducial_{det}_psd.bin")
    fiducial.tofile(bin_out)

    inband = (freqs >= 15.0) & (freqs <= 1024.0)
    meta = {
        "detector": det,
        "num_freq_bins": int(F0),
        "dtype": "float32",
        "byte_order": "little",
        "sample_rate": sr0,
        "delta_f": df0,
        "freq_start": 0.0,
        "freq_end": float(freqs[-1]),
        "num_samples_used": int(ntot),
        "runs_pooled": RUNS,
        "psd_aggregation": "median",
        "worst_case": f"p{PCTL}",
        "blackout_policy": "LocalLineNotch",
        "local_line_notch": {
            "k_depth": K_DEPTH, "window_hz": 8.0, "thresh": 2.0,
            "taper_bins": 1.5, "f_low": 15.0, "f_high": 1024.0,
        },
        "num_line_bins": int(len(line_idx)),
        "line_frac_inband": float(len(line_idx) / max(1, int(inband.sum()))),
    }
    with open(os.path.join(FIDUCIAL_DIR, f"fiducial_{det}_psd.json"), "w") as f:
        json.dump(meta, f, indent=2)

    band = (freqs >= 100.0) & (freqs <= 200.0)
    print(f"[{det}] wrote {bin_out} | {len(line_idx)} line bins "
          f"({100 * len(line_idx) / max(1, int(inband.sum())):.2f}% in-band) | "
          f"median floor@100-200Hz {median[band].mean():.3e}", flush=True)


if __name__ == "__main__":
    for det in DETECTORS:
        build_detector(det)
    print("Done. Combined line-notched fiducial ASDs written to", FIDUCIAL_DIR, flush=True)
