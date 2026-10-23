#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : get_asds.py
Description     : Fiducial / recolour / per-segment ASD estimation.

Created on 2025-12-16 15:44:10

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
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
import h5py
import json
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pycbc import DYN_RANGE_FAC

# LOCAL
from sage.data.primer import NoBlackout
from sage.dsp.inverse_spectrum_truncation import inverse_spectrum_truncation_single

from sage.core.config import get_cfg, get_data_cfg


class EstimateASD:
    """
    Estimate a fiducial ASD by sampling noise from the active noise pipeline.

    Welch returns a PSD; this class square-roots it immediately (see
    :meth:`estimate_raw_asds` and :meth:`estimate_segment_asds`), so every
    quantity produced here -- the per-segment bank, the recolour bank and the
    fiducial itself -- is an amplitude spectral density in strain/sqrt(Hz)
    (~1e-23), not a power spectral density (~1e-46).  That is what the whitener
    and the optimal-SNR integral both want, since each divides by the ASD once.

    The output files keep their historical ``*_psd*`` names so existing data
    releases and run exports stay readable; only the code was renamed.
    """

    def __init__(
        self,
        *,
        detector: str,
        num_samples: int = 200_000,
        asd_method=None,
        blackout_policy=None,
        store_asds_as_hdf5: bool = False,
        store_asds_as_bin: bool = False,
        apply_inverse_spectrum_truncation: bool = False,
        max_filter_len: int | None = None,
        low_frequency_cutoff: float | None = 15.0,
        trunc_method: str = "hann",
        interpolate_asd: bool = False,
        training_sample_length=None,
        asd_smoothener=None,
        **kwargs,
    ):
        # Pull required runtime context
        self.cfg = get_cfg()
        self.data_cfg = get_data_cfg()

        self.detector = detector
        self.num_samples = int(num_samples)
        self.asd_method = asd_method
        self.blackout_policy = blackout_policy or NoBlackout()
        self.store_asds_as_hdf5 = store_asds_as_hdf5
        self.store_asds_as_bin = store_asds_as_bin

        self.apply_ist = apply_inverse_spectrum_truncation
        self.max_filter_len = max_filter_len
        self.low_frequency_cutoff = low_frequency_cutoff
        self.trunc_method = trunc_method

        # Interpolation
        self.interpolate_asd = interpolate_asd
        self.training_sample_length = training_sample_length

        # Spline smoother applied to each ASD before it is written
        self.asd_smoothener = asd_smoothener

        # Sanity checks
        if self.apply_ist:
            if self.max_filter_len is None:
                raise ValueError(
                    "max_filter_len must be set for inverse spectrum truncation"
                )

    @staticmethod
    def _interpolate(
        asd,
        *,
        delta_f_asd: float,
        sample_length: int,
        sample_rate: float,
    ):
        """
        Interpolate an ASD to match the FFT grid of a given sample length.
        Works with NumPy arrays or torch tensors (CPU only).

        Args:
            asd:
                shape (F,) or (..., F)
            delta_f_asd:
                frequency spacing of input ASD
            sample_length:
                time-domain sample length
            sample_rate:
                sampling rate in Hz

        Returns:
            asd_interp:
                shape (..., sample_length//2 + 1)
        """
        # Determine backend
        if "torch" in str(type(asd)):
            asd = asd.detach().cpu().numpy()

        asd = np.asarray(asd)
        orig_shape = asd.shape
        F_asd = orig_shape[-1]

        # Frequency grids
        delta_f_new = sample_rate / sample_length
        F_new = sample_length // 2 + 1

        f_asd = np.arange(F_asd) * delta_f_asd
        f_new = np.arange(F_new) * delta_f_new

        # Reshape to 2D for interpolation
        asd_flat = asd.reshape(-1, F_asd)
        out = np.empty((asd_flat.shape[0], F_new), dtype=asd.dtype)

        # Interpolate
        for i in range(asd_flat.shape[0]):
            out[i] = np.interp(
                f_new,
                f_asd,
                asd_flat[i],
                left=asd_flat[i, 0],
                right=asd_flat[i, -1],
            )

        # Restore shape
        out = out.reshape(*orig_shape[:-1], F_new)

        return out, delta_f_new, f_new

    def taper(self, freqs, asd, asd_floor=3.16e-23):
        """
        Apply a cosine roll-off below the low-frequency cutoff.

        Smoothly transitions the ASD from ``asd_floor`` at DC to the measured
        value at ``low_frequency_cutoff``, imposing C¹ continuity and reducing
        time-domain ringing.

        Parameters
        ----------
        freqs : numpy.ndarray
            Frequency array (Hz).
        asd : numpy.ndarray
            ASD array to be tapered (modified in-place).
        asd_floor : float
            Noise floor applied at DC, in strain/sqrt(Hz) (default ``3.16e-23``).

        Returns
        -------
        numpy.ndarray
            Tapered ASD (same object as *asd*).
        """
        # Tapering down to a noise floor
        # This imposes C1 continuity and reduces ringing effects in TD
        # Make a tapering function below low freq cutoff
        taper = np.ones_like(asd)
        mask = freqs < self.low_frequency_cutoff
        # All values below f_low down to 0.0 Hz
        x = freqs[mask] / self.low_frequency_cutoff  # 0 to 1
        # Cosine roll-off from floor to 1
        taper[mask] = asd_floor + (asd[mask] - asd_floor) * 0.5 * (
            1 - np.cos(np.pi * x)
        )
        # Tapering will take effect and impose a floor
        asd[mask] = taper[mask]
        return asd

    def estimate_raw_asds(self, *, noise_sampler, duration, return_fiducial=False):
        """Run ASD estimation to produce the recolour bank and the fiducial."""
        sample_rate = self.data_cfg.sample_rate

        # Output directories. The ``*_psds`` names are historical -- the files
        # hold ASDs -- and are kept so existing data releases stay readable.
        fiducial_dir = os.path.join(self.cfg.export_dir, "fiducial_psds")
        recolour_dir = os.path.join(self.data_cfg.data_dir, "recolour_psds")

        os.makedirs(fiducial_dir, exist_ok=True)
        os.makedirs(recolour_dir, exist_ok=True)

        n = self.num_samples

        # We stream every ASD straight to disk instead of accumulating the
        # whole (num_samples, F) bank in RAM. The recolour bank is written one
        # row at a time; the fiducial median is computed afterwards by reading
        # the bank back in chunks; and the per-bin maximum that the blackout
        # policies need is tracked incrementally during the sweep.
        bin_path = os.path.join(recolour_dir, f"raw_{self.detector}_psds.bin")

        # If the bank is not meant to be kept, stream to a scratch file that we
        # delete once the median has been computed.
        keep_bin = self.store_asds_as_bin
        stream_path = (
            bin_path
            if keep_bin
            else os.path.join(recolour_dir, f".raw_{self.detector}_psds.tmp.bin")
        )

        freqs = None
        num_freq = None
        delta_f = None
        max_asd = None

        with open(stream_path, "wb") as fh:
            for _ in tqdm(
                range(n),
                desc=f"Estimating recolour ASDs for {self.detector}",
            ):
                # Sample noise sample given duration
                noise = noise_sampler(duration)

                # Welch returns a PSD; take the square root here so everything
                # downstream -- and everything written to disk -- is an ASD.
                pxx = self.asd_method(noise)
                pxx = torch.sqrt(pxx).to(dtype=torch.float32)
                pxx_freqs = self.asd_method.freqs
                pxx_delta_f = 1.0 / (self.asd_method.seg_len * self.asd_method.delta_t)

                if self.apply_ist:
                    raise NotImplementedError("Inverse spectrum truncation removed")

                # Interpolate if requested
                if self.interpolate_asd:
                    pxx, pxx_delta_f, pxx_freqs = EstimateASD._interpolate(
                        asd=pxx,
                        delta_f_asd=pxx_delta_f,
                        sample_length=self.training_sample_length,
                        sample_rate=sample_rate,
                    )

                # Spline smooth the ASD before saving
                if self.asd_smoothener is not None:
                    pxx = self.asd_smoothener.smooth(
                        pxx_freqs, pxx, smooth_factor=0.025 * len(pxx_freqs)
                    )

                # DO NOT do the following although its tempting
                # Kill all values below low frequency cutoff
                # pxx[freqs < self.low_frequency_cutoff] = 1e30
                # This introduces long lasting ringing effects in TD
                # Instead we make a slow taper
                pxx = self.taper(pxx_freqs, pxx)

                # Stream this ASD straight to disk (row-major, float32)
                pxx = np.ascontiguousarray(pxx, dtype=np.float32)
                fh.write(pxx.tobytes())

                # Track the per-bin maximum (exact and order-independent) for
                # the blackout policy, and capture the frequency grid once (it
                # is identical on every sweep).
                if max_asd is None:
                    max_asd = pxx.copy()
                    freqs = np.asarray(pxx_freqs, dtype=np.float64)
                    num_freq = pxx.shape[0]
                    delta_f = float(pxx_delta_f)
                else:
                    np.maximum(max_asd, pxx, out=max_asd)

        # Sidecar metadata for the recolour bank (consumed by recolour.py)
        if self.store_asds_as_bin:
            self._write_recolour_bank_meta(
                recolour_dir, n, num_freq, freqs, sample_rate
            )

        # Optional gzip-HDF5 archival of the bank (chunked, low memory)
        if self.store_asds_as_hdf5:
            self._save_raw_asds_hdf5(
                stream_path, recolour_dir, n, num_freq, freqs, sample_rate
            )

        # Compute the median ASD by streaming the on-disk bank in chunks, so the
        # full (num_samples, F) array is never resident in memory.
        bank = np.memmap(
            stream_path, dtype=np.float32, mode="r", shape=(n, num_freq)
        )
        median_asd = self._aggregate_asds(bank)
        del bank

        # Drop the scratch bank if we were not asked to keep it
        if not keep_bin:
            os.remove(stream_path)

        # Compute fiducial ASD; blackout policies need only the per-bin maximum
        fiducial_asd, blackout_idxs = self.blackout_policy.apply(median_asd, max_asd)

        # Saving fiducial ASD in export_dir of run
        self._save_fiducial_asd(
            fiducial_asd,
            freqs,
            blackout_idxs,
            fiducial_dir,
            sample_rate,
        )

        if return_fiducial:
            return freqs, fiducial_asd

    def _aggregate_asds(self, bank):
        # Median of medians, reading the on-disk bank one chunk at a time so the
        # full (num_samples, F) array is never resident in memory. ``bank`` is
        # any row-sliceable array (np.memmap / h5py dataset).
        num_asds = bank.shape[0]
        chunks = np.array_split(np.arange(num_asds), max(1, num_asds // 10_000))
        medians = [
            np.median(np.asarray(bank[idx[0] : idx[-1] + 1]), axis=0)
            for idx in chunks
        ]

        median_asd = np.median(medians, axis=0)
        return median_asd

    @staticmethod
    def _to_float(x):
        if torch.is_tensor(x):
            return float(x.item())
        return float(x)

    def _write_recolour_bank_meta(self, save_dir, num_asds, num_freq, freqs, sample_rate):
        # The bank .bin is streamed to disk during estimate_raw_asds; here we
        # only emit the sidecar JSON the recolour module reads. The ``psd``
        # spellings in the key names are the on-disk contract -- left alone so
        # sidecars written before and after this rename stay interchangeable.
        meta_path = os.path.join(save_dir, f"raw_{self.detector}_psds.json")
        meta = {
            "detector": self.detector,
            "num_psds": num_asds,
            "num_freq_bins": num_freq,
            "dtype": "float32",
            "byte_order": "little",
            "layout": "row-major",
            "sample_rate": sample_rate,
            "delta_f": EstimateASD._to_float(freqs[1] - freqs[0]),
            "freq_start": EstimateASD._to_float(freqs[0]),
            "freq_end": EstimateASD._to_float(freqs[-1]),
            "psd_method": self.asd_method.__class__.__name__,
            "apply_inverse_spectrum_truncation": self.apply_ist,
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "max_filter_len": self.max_filter_len,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _save_raw_asds_hdf5(self, bin_path, save_dir, num_asds, num_freq, freqs, sample_rate):
        # Archive the streamed bank as a gzip-compressed HDF5 dataset, copied in
        # chunks so the full bank is never held in memory at once.
        hdf5_path = os.path.join(save_dir, f"raw_{self.detector}_psds.h5")
        bank = np.memmap(
            bin_path, dtype=np.float32, mode="r", shape=(num_asds, num_freq)
        )
        step = 1000
        with h5py.File(hdf5_path, "w") as hf:
            dset = hf.create_dataset(
                "psds",
                shape=(num_asds, num_freq),
                dtype="float32",
                chunks=(min(step, num_asds), num_freq),
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            for start in range(0, num_asds, step):
                end = min(start + step, num_asds)
                dset[start:end] = np.asarray(bank[start:end])
            hf.create_dataset("freqs", data=freqs)
            hf.attrs["sample_rate"] = sample_rate
        del bank

    def _save_fiducial_asd(
        self,
        asd,
        freqs,
        blackout_idxs,
        fiducial_dir,
        sample_rate,
    ):
        # Fiducial ASDs saved in export directory
        if self.store_asds_as_hdf5:
            hdf5_path = os.path.join(fiducial_dir, f"fiducial_{self.detector}_psd.h5")

            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)

            with h5py.File(hdf5_path, "w") as hf:
                hf.create_dataset(
                    "psd",
                    data=asd,
                    compression="gzip",
                    compression_opts=9,
                    shuffle=True,
                )

                hf.create_dataset("freqs", data=freqs)
                # Handles when blackout_idxs is None
                hf.create_dataset("blackout_indices", data=blackout_idxs)

                hf.attrs.update(
                    {
                        "detector": self.detector,
                        "delta_f": freqs[1] - freqs[0],
                        "num_freq_bins": len(asd),
                        "freq_start": freqs[0],
                        "freq_end": freqs[-1],
                        "blackout_policy": self.blackout_policy.__class__.__name__,
                        "num_samples_used": self.num_samples,
                        "sample_rate": sample_rate,
                        "psd_aggregation": "median",
                        "blackout_indices": (
                            blackout_idxs.tolist()
                            if blackout_idxs is not None
                            else None
                        ),
                        "low_frequency_cutoff": self.low_frequency_cutoff,
                        "max_filter_len": self.max_filter_len,
                    }
                )

        elif self.store_asds_as_bin:
            bin_path = os.path.join(fiducial_dir, f"fiducial_{self.detector}_psd.bin")
            np.asarray(asd, dtype=np.float32).tofile(bin_path)

            meta = {
                "detector": self.detector,
                "num_freq_bins": len(asd),
                "dtype": "float32",
                "byte_order": "little",
                "sample_rate": sample_rate,
                "delta_f": EstimateASD._to_float(freqs[1] - freqs[0]),
                "freq_start": EstimateASD._to_float(freqs[0]),
                "freq_end": EstimateASD._to_float(freqs[-1]),
                "num_samples_used": self.num_samples,
                "psd_aggregation": "median",
                "blackout_policy": self.blackout_policy.__class__.__name__,
                "blackout_indices": (
                    blackout_idxs.tolist() if blackout_idxs is not None else None
                ),
                "apply_inverse_spectrum_truncation": self.apply_ist,
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "max_filter_len": self.max_filter_len,
            }

            meta_path = os.path.join(fiducial_dir, f"fiducial_{self.detector}_psd.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    def estimate_segment_asds(self, *, noise_segments_file):
        """
        Compute the Welch ASD for each noise segment in a bin file.

        Args:
            noise_segments_file: path to noise .bin file
            output_dir: directory to write the ASD bin + metadata
        """
        noise_segments_file = Path(noise_segments_file)
        output_dir = Path(self.data_cfg.data_dir) / "segment_psds"
        output_dir.mkdir(parents=True, exist_ok=True)

        meta_path = (
            noise_segments_file.parent / f"{noise_segments_file.stem}_segments.json"
        )
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        with open(meta_path, "r") as f:
            seg_meta = json.load(f)

        # dtype
        dt = np.dtype(seg_meta[0]["dtype"]).newbyteorder(seg_meta[0]["endianness"])
        mm = np.memmap(noise_segments_file, dtype=dt, mode="r")

        # NOTE: filename is detector-based (no run label) to match the
        # consumer in sage/data/noise/recolour.py, which loads segment ASDs as
        # ``data_{det}_psds.bin``. Per-run separation is provided by data_dir.
        asd_bin_path = output_dir / f"data_{self.detector}_psds.bin"
        asd_meta_path = output_dir / f"data_{self.detector}_psds_segments.json"

        asd_meta = []
        asd_cursor = 0

        with open(asd_bin_path, "wb") as asd_fh:
            for seg in tqdm(seg_meta, desc="Computing ASDs per segment"):
                start = seg["sample_start_idx"]
                nsamp = seg["nsamples"]

                data = np.array(
                    mm[start : start + nsamp],
                    dtype=np.float32,
                    copy=True,
                )
                data /= DYN_RANGE_FAC

                ts = torch.from_numpy(data)

                # Welch gives a PSD; square-root it so the bank on disk is
                # an ASD, matching the fiducial and what recolour expects.
                asd = self.asd_method(ts).cpu().numpy()
                asd = np.sqrt(asd).astype(np.float32)
                freqs = self.asd_method.freqs
                delta_f = 1.0 / (self.asd_method.seg_len * self.asd_method.delta_t)

                # Apply inverse spectrum truncation
                if self.apply_ist:
                    raise NotImplementedError("Inverse spectrum truncation removed")
                    asd = torch.from_numpy(asd).to(torch.float64)
                    asd = inverse_spectrum_truncation_single(
                        psd=asd,
                        max_filter_len=self.max_filter_len,
                        low_frequency_cutoff=self.low_frequency_cutoff,
                        delta_f=delta_f,
                        trunc_method=self.trunc_method,
                    )
                    asd = asd.cpu().numpy()

                # Interpolate if requested
                if self.interpolate_asd:
                    asd, delta_f, freqs = EstimateASD._interpolate(
                        asd=asd,
                        delta_f_asd=delta_f,
                        sample_length=self.training_sample_length,
                        sample_rate=self.data_cfg.sample_rate,
                    )

                # Spline smooth the ASD before saving.
                # Smoothing strength is domain dependent: 0.004 * len(freqs)
                # would be right for a PSD, 0.001 for the ASD we have here.
                if self.asd_smoothener is not None:
                    asd = self.asd_smoothener.smooth(
                        freqs, asd, smooth_factor=0.001 * len(freqs)
                    )

                # DO NOT do the following although its tempting
                # Kill all values below low frequency cutoff
                # asd[freqs < self.low_frequency_cutoff] = 1e30
                # This introduces long lasting ringing effects in TD
                # Instead we make a slow taper
                asd = self.taper(freqs, asd)

                nbytes = asd.nbytes
                asd_fh.write(asd.tobytes())

                asd_meta.append(
                    {
                        "noise_segment_index": seg["segment_index"],
                        "gps_start": seg["gps_start"],
                        "gps_end": seg["gps_end"],
                        "sample_rate": seg["sample_rate"],
                        "psd_len": asd.shape[0],
                        "byte_offset": asd_cursor,
                        "byte_length": nbytes,
                        "delta_f": delta_f,
                        "seg_len": self.asd_method.seg_len,
                        "seg_stride": self.asd_method.seg_stride,
                        "window": "hann",
                        "inverse_spectrum_truncation": 1 if self.apply_ist else 0,
                        "max_filter_len": self.max_filter_len,
                        "low_frequency_cutoff": self.low_frequency_cutoff,
                        "interpolation": 1 if self.interpolate_asd else 0,
                    }
                )

                asd_cursor += nbytes

        with open(asd_meta_path, "w") as f:
            json.dump(asd_meta, f, indent=2)
