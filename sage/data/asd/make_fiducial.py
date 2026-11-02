"""
Fiducial ASDs: built from pooled recolour banks, or resampled onto a different run's FFT grid.

A fiducial pools the recolour banks of one or more observing runs: the exact
per-bin median is the broadband floor, and spectral lines found by Thomson's
F-test in stretches of each run's strain are excised with smooth edges
(:mod:`sage.data.asd.lines`): infinite ASD, so zero whitening weight.

Resampling (``regenerate_fiducials``) uses the interpolation ``EstimateASD`` uses
at production (:meth:`EstimateASD._interpolate` -> ``np.interp``, linear,
edge-clamped). It does NOT add real spectral resolution -- that is set by the
Welch window used at production -- it only lands the real ASD on the analysis
grid. The whitener (:func:`sage.data.asd.read_asds.read_fiducial_asds`) loads it
verbatim, so the on-disk file must already be at the run's grid.

The ``fiducial_{det}_psd.{bin,json}`` filenames are historical; the contents are
ASDs in strain/sqrt(Hz).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sage.data.asd.lines import excision_weight, line_bins
from sage.data.primer.get_asds import EstimateASD

BLOCK = 2048
ROW_CHUNK = 50_000


def pooled_median(banks: list) -> np.ndarray:
    """
    Exact per-bin median ASD over every row of pooled recolour banks.

    Parameters
    ----------
    banks : list of numpy.ndarray
        Recolour banks, each ``(rows, bins)`` float32, usually memory-mapped.

    Returns
    -------
    numpy.ndarray
        Median ASD, float64.
    """
    # Bins one block at a time, rows read in sequential chunks: a strided column
    # read of a whole bank is far slower, and the pooled banks do not fit in memory.
    total, bins = sum(len(bank) for bank in banks), banks[0].shape[1]
    median = np.empty(bins, dtype=np.float64)
    for first in range(0, bins, BLOCK):
        last = min(bins, first + BLOCK)
        pooled = np.empty((total, last - first), dtype=np.float32)
        offset = 0
        for bank in banks:
            for row in range(0, len(bank), ROW_CHUNK):
                end = min(len(bank), row + ROW_CHUNK)
                pooled[offset + row:offset + end] = bank[row:end, first:last]
            offset += len(bank)
        # q=50 (linear) equals np.median for an even row count; q as a
        # list keeps the float32 rows' interpolation in float64.
        median[first:last] = np.percentile(
            pooled, [50.0], axis=0, overwrite_input=True
        )[0]
        del pooled
    return median


def pooled_fiducial(
    banks: list, detections: np.ndarray, stretches: int, nw: float
) -> tuple:
    """
    Fiducial ASD of one detector: pooled median, lines excised.

    The stretches are ``1 / delta_f`` long, as the banks' ASDs, so the
    excised band and its edge are NW bins each.

    Parameters
    ----------
    banks : list of numpy.ndarray
        Recolour banks, each ``(rows, bins)`` float32, usually memory-mapped.
    detections : numpy.ndarray of int
        F-test detections per bin over the pooled runs' stretches
        (:func:`sage.data.asd.lines.line_detections`).
    stretches : int
        Stretches the detections come from.
    nw : float
        Time-bandwidth product NW of the F-test.

    Returns
    -------
    numpy.ndarray
        Fiducial ASD, float32, infinite at excised bins.
    numpy.ndarray
        Indices of the line bins.
    """
    median = pooled_median(banks)
    bins = len(median)
    lines = line_bins(detections, stretches, 1 / bins)
    weight = excision_weight(lines, bins, round(nw))
    with np.errstate(divide="ignore"):
        fiducial = median / weight
    return fiducial.astype(np.float32), lines


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
