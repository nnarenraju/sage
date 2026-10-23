#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_fiducial.py -- combined-run, line-notched fiducial ASDs (one PROFILE per build).

Builds the production fiducial ASDs shared by every run, from the
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

A fiducial pools exactly the runs the networks using it will SEE -- training and
validation both -- because pooling the evaluated run is what keeps its lines
notched in the whitening reference, and pooling only the relevant epoch is what
keeps the broadband floor (and hence the fiducial-defined SNR label) matched to
it. Networks over different run sets therefore need different fiducials, one per
PROFILE below.

Writes ``fiducial_{det}_psd.bin`` + ``.json`` to that profile's ``FIDUCIAL_DIR``
(absolute, shared). Run once on a CPU node before training; pass detectors to
restrict it:

    python runs/build_fiducial.py                      # every detector in the profile
    python runs/build_fiducial.py H1 L1                # just the HL networks
    SAGE_FIDUCIAL_PROFILE=o3o4b python runs/build_fiducial.py    # the Virgo profile
"""

import os
import json

import numpy as np

from sage.utils.servers import get_server
from sage.data.asd.blackout import LocalLineNotch

# Combined O3+O4 fiducial location -- MUST match config_base.FIDUCIAL_DIR in the
# runs that use it. Written to a NEW directory: fiducial_psds_o3ab is the live
# O3a/O3b production reference and must not be overwritten under running jobs.
# Fiducial profiles, selected by SAGE_FIDUCIAL_PROFILE (default "o4"). Each maps
# an absolute output dir to the per-detector run pools it aggregates; a run appears
# only where it actually has that detector (Virgo did not observe in O4a, so
# data_release_o4a carries H1/L1 banks only).
#
#   o4     H1/L1 over O4a+O4b. The whitening reference for the two O4 HL production
#          networks: each trains on one O4 run and validates on the other, so the
#          pool is exactly the two runs they see. The direct analogue of the O3
#          production runs sharing a pooled O3a+O3b fiducial.
#
#   o3o4b  H1/L1/V1 over O3a+O3b+O4b, for the DEFERRED O4b Virgo networks
#          (HV/LV/HLV). Those have to train on O3a+O3b -- O4a carries no Virgo --
#          and validate on O4b, so all three runs must be in the pool.
PROFILES = {
    "o4": (
        "/work/nagarajan/sage_runs/fiducial_psds_o4",
        {"H1": ["O4a", "O4b"], "L1": ["O4a", "O4b"]},
    ),
    "o3o4b": (
        "/work/nagarajan/sage_runs/fiducial_psds_o3o4b",
        {"H1": ["O3a", "O3b", "O4b"],
         "L1": ["O3a", "O3b", "O4b"],
         "V1": ["O3a", "O3b", "O4b"]},
    ),
}
_PROFILE = os.environ.get("SAGE_FIDUCIAL_PROFILE", "o4")
if _PROFILE not in PROFILES:
    raise SystemExit(
        f"unknown SAGE_FIDUCIAL_PROFILE {_PROFILE!r}; choose from {sorted(PROFILES)}"
    )
FIDUCIAL_DIR, RUNS_BY_DETECTOR = PROFILES[_PROFILE]
DETECTORS = list(RUNS_BY_DETECTOR)

# The O3 releases keep their derived banks under <data_release>/<run>_dataset/data_dir
# (Server.data_dir); the O4 releases use a flat <data_release_<run>>/data_dir. Only
# the exceptions are listed -- anything absent here falls back to Server.data_dir.
_DATA_DIR_OVERRIDES = {
    "O4a": "/work/nagarajan/data_release_o4a/data_dir",
    "O4b": "/work/nagarajan/data_release_o4b/data_dir",
}

K_DEPTH = 4.0            # fiducial at a line bin = K_DEPTH * worst  -> whitened line <= ~1/K
PCTL = 99.9             # robust worst-case percentile (NOT the true max)
BLOCK = 2048            # frequency-bin block for the exact column-wise percentile
ROWCHUNK = 50_000       # segments per sequential read when filling a block buffer


def _paths(run, det):
    base = _DATA_DIR_OVERRIDES.get(run) or get_server().data_dir(run)
    rec = os.path.join(base, "recolour_psds")
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
    runs = RUNS_BY_DETECTOR[det]
    banks, F0, df0, sr0 = [], None, None, None
    for run in runs:
        mm, n, F, df, sr = _open_bank(run, det)
        if F0 is None:
            F0, df0, sr0 = F, df, sr
        elif (F, df) != (F0, df0):
            raise RuntimeError(
                f"{det}: grid mismatch {run} (F={F}, df={df}) vs (F={F0}, df={df0})"
            )
        banks.append((mm, n))
    ntot = sum(n for _, n in banks)
    print(f"[{det}] pooling " + " + ".join(f"{run}:{n}" for run, (_, n) in zip(runs, banks))
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
        "runs_pooled": runs,
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
    import sys

    wanted = sys.argv[1:] or DETECTORS
    unknown = [d for d in wanted if d not in RUNS_BY_DETECTOR]
    if unknown:
        raise SystemExit(f"unknown detector(s) {unknown}; known: {DETECTORS}")
    for det in wanted:
        build_detector(det)
    print(f"Done. Profile {_PROFILE!r} line-notched fiducial ASDs written to",
          FIDUCIAL_DIR, flush=True)
