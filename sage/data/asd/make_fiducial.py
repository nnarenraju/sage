"""
Regenerate fiducial ASDs onto a different run's FFT grid.

A fiducial ASD is stored per-detector at the resolution it was produced at
(``EstimateASD(interpolate_asd=True, training_sample_length=...)``).  To reuse a
run's REAL fiducial for a longer/shorter analysis segment, resample it onto the
target signal grid via the SAME tested interpolation ``EstimateASD`` uses at
production (:meth:`EstimateASD._interpolate` -> ``np.interp``, linear, edge-clamped).

This is a one-time data-prep step (output kept on disk): the smooth broadband ASD
is resampled to the target ``delta_f``.  It does NOT add real spectral resolution
— that is set by the Welch window used at production — it only lands the real ASD
on the analysis grid.  The whitener (:func:`sage.data.asd.read_asds.get_fiducial_asds`)
then loads it verbatim, so the on-disk file must already be at the run's grid.

The ``fiducial_{det}_psd.{bin,json}`` filenames are historical; the contents are
ASDs in strain/sqrt(Hz).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sage.data.primer.get_asds import EstimateASD


def interpolate_fiducial(asd, src_delta_f, target_sample_length, sample_rate):
    """
    Resample a per-detector fiducial ASD onto the rFFT grid of
    ``target_sample_length`` (``target_sample_length // 2 + 1`` bins).

    Thin wrapper over :meth:`EstimateASD._interpolate` (linear, edge-clamped).
    Returns ``(asd_target_float32, delta_f_target)``.
    """
    out, delta_f_new, _ = EstimateASD._interpolate(
        np.asarray(asd, dtype=np.float64),
        delta_f_asd=float(src_delta_f),
        sample_length=int(target_sample_length),
        sample_rate=float(sample_rate),
    )
    return out.astype(np.float32), float(delta_f_new)


def regenerate_fiducials(
    src_dir,
    dst_dir,
    detectors,
    target_sample_length,
    sample_rate,
    note: str = "",
):
    """
    Load each detector's fiducial from ``src_dir``, interpolate to the target
    grid, and write ``.bin`` + ``.json`` to ``dst_dir``.

    Parameters
    ----------
    src_dir, dst_dir : path-like
        Directories holding ``fiducial_{det}_psd.{bin,json}``.
    detectors : list[str]
        e.g. ``["H1", "L1"]``.
    target_sample_length : int
        Padded segment length in samples of the target run
        (``data_cfg.padded_length_in_nsamples``).
    sample_rate : float
    note : str
        Free-text note stored in the output json.

    Returns
    -------
    dict[str, tuple[int, float]]
        ``det -> (n_freq_target, delta_f_target)``.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_freq_target = int(target_sample_length) // 2 + 1

    result = {}
    for det in detectors:
        meta = json.loads((src_dir / f"fiducial_{det}_psd.json").read_text())
        n_freq_src = int(meta.get("num_freq_bins") or meta.get("n_fd"))
        src_delta_f = float(
            meta.get("delta_f")
            or (sample_rate / (2.0 * (n_freq_src - 1)))
        )
        src_asd = np.fromfile(src_dir / f"fiducial_{det}_psd.bin", dtype=np.float32)
        if src_asd.shape[-1] != n_freq_src:
            raise ValueError(
                f"{det}: fiducial .bin has {src_asd.shape[-1]} bins, json says {n_freq_src}"
            )

        out, delta_f_new = interpolate_fiducial(
            src_asd, src_delta_f, target_sample_length, sample_rate
        )
        assert out.shape[-1] == n_freq_target, (out.shape[-1], n_freq_target)

        out.tofile(dst_dir / f"fiducial_{det}_psd.bin")
        (dst_dir / f"fiducial_{det}_psd.json").write_text(
            json.dumps(
                {
                    "detector": det,
                    "n_fd": n_freq_target,
                    "dtype": "float32",
                    "padded_length": int(target_sample_length),
                    "sample_rate": float(sample_rate),
                    "delta_f": delta_f_new,
                    "source_bin": str(src_dir / f"fiducial_{det}_psd.bin"),
                    "source_delta_f": src_delta_f,
                    "source_n_freq": n_freq_src,
                    "note": note
                    or (f"REAL fiducial ASD resampled (linear) from delta_f={src_delta_f:g} "
                        f"to {delta_f_new:g}; broadband real content, resolution set at source."),
                },
                indent=2,
            )
        )
        result[det] = (n_freq_target, delta_f_new)
    return result
