#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rebuild_fiducial.py

Rebuild the fiducial PSDs for the O3b run from the recolour raw PSD banks that
are already on disk. The original PSD-generation run (Jun 2) crashed at the very
last step writing the fiducial files because run_export/fiducial_psds/ did not
exist (the makedirs guard was added later). The expensive raw banks completed
fine, so the fiducial is just the median-of-medians of those banks (NoBlackout
policy) -- no noise re-sampling needed.

This replicates EstimatePSD._aggregate_psds + _save_fiducial_psd exactly.
"""

import os
import json

import numpy as np

from sage.utils.servers import get_server
RECOLOUR_DIR = os.path.join(get_server().data_dir("O3b"), "recolour_psds")
FIDUCIAL_DIR = os.path.join(os.path.dirname(__file__), "run_export", "fiducial_psds")

DETECTORS = ["H1", "L1", "V1"]


def _aggregate_psds(bank):
    # Median of medians, reading the on-disk bank one chunk at a time.
    # Verbatim copy of EstimatePSD._aggregate_psds (get_psds.py:321).
    num_psds = bank.shape[0]
    chunks = np.array_split(np.arange(num_psds), max(1, num_psds // 10_000))
    medians = [
        np.median(np.asarray(bank[idx[0] : idx[-1] + 1]), axis=0) for idx in chunks
    ]
    return np.median(medians, axis=0)


def rebuild_detector(det):
    bank_path = os.path.join(RECOLOUR_DIR, f"raw_{det}_psds.bin")
    meta_path = os.path.join(RECOLOUR_DIR, f"raw_{det}_psds.json")

    with open(meta_path, "r") as f:
        bank_meta = json.load(f)

    n = int(bank_meta["num_psds"])
    num_freq = int(bank_meta["num_freq_bins"])
    delta_f = float(bank_meta["delta_f"])
    sample_rate = float(bank_meta["sample_rate"])

    expected_bytes = n * num_freq * 4
    actual_bytes = os.path.getsize(bank_path)
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{det}: bank size {actual_bytes} != expected {expected_bytes} "
            f"(n={n}, num_freq={num_freq}). Bank may be incomplete."
        )

    print(f"[{det}] reading bank ({n} x {num_freq}) ...", flush=True)
    bank = np.memmap(bank_path, dtype=np.float32, mode="r", shape=(n, num_freq))
    median_psd = _aggregate_psds(bank)
    del bank

    # NoBlackout policy: pass-through, empty blackout index array.
    fiducial_psd = np.asarray(median_psd, dtype=np.float32)
    freqs = np.arange(num_freq, dtype=np.float64) * delta_f

    os.makedirs(FIDUCIAL_DIR, exist_ok=True)

    bin_out = os.path.join(FIDUCIAL_DIR, f"fiducial_{det}_psd.bin")
    fiducial_psd.tofile(bin_out)

    meta = {
        "detector": det,
        "num_freq_bins": int(len(fiducial_psd)),
        "dtype": "float32",
        "byte_order": "little",
        "sample_rate": sample_rate,
        "delta_f": delta_f,
        "freq_start": float(freqs[0]),
        "freq_end": float(freqs[-1]),
        "num_samples_used": n,
        "psd_aggregation": "median",
        "blackout_policy": "NoBlackout",
        "blackout_indices": None,
        "apply_inverse_spectrum_truncation": bool(
            bank_meta.get("apply_inverse_spectrum_truncation", False)
        ),
        "low_frequency_cutoff": bank_meta.get("low_frequency_cutoff", 15.0),
        "max_filter_len": bank_meta.get("max_filter_len", 4096),
        "rebuilt_from": os.path.basename(bank_path),
    }
    meta_out = os.path.join(FIDUCIAL_DIR, f"fiducial_{det}_psd.json")
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[{det}] wrote {bin_out} "
        f"({len(fiducial_psd)} bins, min={fiducial_psd.min():.3e}, "
        f"max={fiducial_psd.max():.3e})",
        flush=True,
    )


if __name__ == "__main__":
    for det in DETECTORS:
        rebuild_detector(det)
    print("Done. Fiducial PSDs rebuilt in", FIDUCIAL_DIR, flush=True)
