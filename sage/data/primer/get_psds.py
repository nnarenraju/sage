#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : get_psds.py
Description     : Short description of the file

Created on 2025-12-16 15:44:10

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2025, ProjectName
__license__       = MIT Licence
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


class EstimatePSD:
    """
    Estimate a fiducial PSD by sampling noise from the active noise pipeline.
    """

    def __init__(
        self,
        *,
        detector: str,
        num_samples: int = 200_000,
        psd_method=None,
        blackout_policy=None,
        store_psds_as_hdf5: bool = False,
        store_psds_as_bin: bool = False,
        apply_inverse_spectrum_truncation: bool = False,
        max_filter_len: int | None = None,
        low_frequency_cutoff: float | None = 15.0,
        trunc_method: str = "hann",
        interpolate_psd: bool = False,
        training_sample_length=None,
        psd_smoothener=None,
        **kwargs,
    ):
        # Pull required runtime context
        self.cfg = get_cfg()
        self.data_cfg = get_data_cfg()

        self.detector = detector
        self.num_samples = int(num_samples)
        self.psd_method = psd_method
        self.blackout_policy = blackout_policy or NoBlackout()
        self.store_psds_as_hdf5 = store_psds_as_hdf5
        self.store_psds_as_bin = store_psds_as_bin

        self.apply_ist = apply_inverse_spectrum_truncation
        self.max_filter_len = max_filter_len
        self.low_frequency_cutoff = low_frequency_cutoff
        self.trunc_method = trunc_method

        # Interpolation
        self.interpolate_psd = interpolate_psd
        self.training_sample_length = training_sample_length

        # Smoothen PSD
        self.psd_smoothener = psd_smoothener

        # Sanity checks
        if self.apply_ist:
            if self.max_filter_len is None:
                raise ValueError(
                    "max_filter_len must be set for inverse spectrum truncation"
                )

    @staticmethod
    def _interpolate(
        psd,
        *,
        delta_f_psd: float,
        sample_length: int,
        sample_rate: float,
    ):
        """
        Interpolate PSD to match FFT grid of a given sample length.
        Works with NumPy arrays or torch tensors (CPU only).

        Args:
            psd:
                shape (F,) or (..., F)
            delta_f_psd:
                frequency spacing of input PSD
            sample_length:
                time-domain sample length
            sample_rate:
                sampling rate in Hz

        Returns:
            psd_interp:
                shape (..., sample_length//2 + 1)
        """
        # Determine backend
        if "torch" in str(type(psd)):
            psd = psd.detach().cpu().numpy()

        psd = np.asarray(psd)
        orig_shape = psd.shape
        F_psd = orig_shape[-1]

        # Frequency grids
        delta_f_new = sample_rate / sample_length
        F_new = sample_length // 2 + 1

        f_psd = np.arange(F_psd) * delta_f_psd
        f_new = np.arange(F_new) * delta_f_new

        # Reshape to 2D for interpolation
        psd_flat = psd.reshape(-1, F_psd)
        out = np.empty((psd_flat.shape[0], F_new), dtype=psd.dtype)

        # Interpolate
        for i in range(psd_flat.shape[0]):
            out[i] = np.interp(
                f_new,
                f_psd,
                psd_flat[i],
                left=psd_flat[i, 0],
                right=psd_flat[i, -1],
            )

        # Restore shape
        out = out.reshape(*orig_shape[:-1], F_new)

        return out, delta_f_new, f_new

    def taper(self, freqs, psd, psd_floor=3.16e-23):
        """
        Apply a cosine roll-off below the low-frequency cutoff.

        Smoothly transitions the PSD from ``psd_floor`` at DC to the measured
        value at ``low_frequency_cutoff``, imposing C¹ continuity and reducing
        time-domain ringing.

        Parameters
        ----------
        freqs : numpy.ndarray
            Frequency array (Hz).
        psd : numpy.ndarray
            PSD array to be tapered (modified in-place).
        psd_floor : float
            Noise floor value applied at DC (default ``3.16e-23``).

        Returns
        -------
        numpy.ndarray
            Tapered PSD (same object as *psd*).
        """
        # Tapering down to a noise floor
        # This imposes C1 continuity and reduces ringing effects in TD
        # Make a tapering function below low freq cutoff
        taper = np.ones_like(psd)
        mask = freqs < self.low_frequency_cutoff
        # All values below f_low down to 0.0 Hz
        x = freqs[mask] / self.low_frequency_cutoff  # 0 to 1
        # Cosine roll-off from floor to 1
        taper[mask] = psd_floor + (psd[mask] - psd_floor) * 0.5 * (
            1 - np.cos(np.pi * x)
        )
        # Tapering will take effect and impose a floor
        psd[mask] = taper[mask]
        return psd

    def estimate_raw_psds(self, *, noise_sampler, duration, return_fiducial=False):
        """Run PSD estimation to get recolour and fiducial psds"""
        sample_rate = self.data_cfg.sample_rate

        # Save directories for PSDs and Fiducial PSDs
        fiducial_dir = os.path.join(self.cfg.export_dir, "fiducial_psds")
        recolour_dir = os.path.join(self.data_cfg.data_dir, "recolour_psds")

        os.makedirs(fiducial_dir, exist_ok=True)
        os.makedirs(recolour_dir, exist_ok=True)

        n = self.num_samples

        # We stream every PSD straight to disk instead of accumulating the
        # whole (num_samples, F) bank in RAM. The recolour bank is written one
        # row at a time; the fiducial median is computed afterwards by reading
        # the bank back in chunks; and the per-bin maximum that the blackout
        # policies need is tracked incrementally during the sweep.
        bin_path = os.path.join(recolour_dir, f"raw_{self.detector}_psds.bin")

        # If the bank is not meant to be kept, stream to a scratch file that we
        # delete once the median has been computed.
        keep_bin = self.store_psds_as_bin
        stream_path = (
            bin_path
            if keep_bin
            else os.path.join(recolour_dir, f".raw_{self.detector}_psds.tmp.bin")
        )

        freqs = None
        num_freq = None
        delta_f = None
        max_psd = None

        with open(stream_path, "wb") as fh:
            for _ in tqdm(
                range(n),
                desc=f"Estimating recolour PSDs for {self.detector}",
            ):
                # Sample noise sample given duration
                noise = noise_sampler(duration)

                # Compute PSD using the Welch method
                pxx = self.psd_method(noise)
                pxx = torch.sqrt(pxx).to(dtype=torch.float32)
                pxx_freqs = self.psd_method.freqs
                pxx_delta_f = 1.0 / (self.psd_method.seg_len * self.psd_method.delta_t)

                if self.apply_ist:
                    raise NotImplementedError("Inverse spectrum truncation removed")

                # Interpolate if requested
                if self.interpolate_psd:
                    pxx, pxx_delta_f, pxx_freqs = EstimatePSD._interpolate(
                        psd=pxx,
                        delta_f_psd=pxx_delta_f,
                        sample_length=self.training_sample_length,
                        sample_rate=sample_rate,
                    )

                # Spline smooth the PSD before saving
                if self.psd_smoothener is not None:
                    pxx = self.psd_smoothener.smooth(
                        pxx_freqs, pxx, smooth_factor=0.025 * len(pxx_freqs)
                    )

                # DO NOT do the following although its tempting
                # Kill all values below low frequency cutoff
                # pxx[freqs < self.low_frequency_cutoff] = 1e30
                # This introduces long lasting ringing effects in TD
                # Instead we make a slow taper
                pxx = self.taper(pxx_freqs, pxx)

                # Stream this PSD straight to disk (row-major, float32)
                pxx = np.ascontiguousarray(pxx, dtype=np.float32)
                fh.write(pxx.tobytes())

                # Track the per-bin maximum (exact and order-independent) for
                # the blackout policy, and capture the frequency grid once (it
                # is identical on every sweep).
                if max_psd is None:
                    max_psd = pxx.copy()
                    freqs = np.asarray(pxx_freqs, dtype=np.float64)
                    num_freq = pxx.shape[0]
                    delta_f = float(pxx_delta_f)
                else:
                    np.maximum(max_psd, pxx, out=max_psd)

        # Sidecar metadata for the recolour bank (consumed by recolour.py)
        if self.store_psds_as_bin:
            self._write_recolour_bank_meta(
                recolour_dir, n, num_freq, freqs, sample_rate
            )

        # Optional gzip-HDF5 archival of the bank (chunked, low memory)
        if self.store_psds_as_hdf5:
            self._save_raw_psds_hdf5(
                stream_path, recolour_dir, n, num_freq, freqs, sample_rate
            )

        # Compute the median PSD by streaming the on-disk bank in chunks, so the
        # full (num_samples, F) array is never resident in memory.
        bank = np.memmap(
            stream_path, dtype=np.float32, mode="r", shape=(n, num_freq)
        )
        median_psd = self._aggregate_psds(bank)
        del bank

        # Drop the scratch bank if we were not asked to keep it
        if not keep_bin:
            os.remove(stream_path)

        # Compute fiducial PSD; blackout policies need only the per-bin maximum
        fiducial_psd, blackout_idxs = self.blackout_policy.apply(median_psd, max_psd)

        # Saving fiducial PSD in export_dir of run
        self._save_fiducial_psd(
            fiducial_psd,
            freqs,
            blackout_idxs,
            fiducial_dir,
            sample_rate,
        )

        if return_fiducial:
            return freqs, fiducial_psd

    def _aggregate_psds(self, bank):
        # Median of medians, reading the on-disk bank one chunk at a time so the
        # full (num_samples, F) array is never resident in memory. ``bank`` is
        # any row-sliceable array (np.memmap / h5py dataset).
        num_psds = bank.shape[0]
        chunks = np.array_split(np.arange(num_psds), max(1, num_psds // 10_000))
        medians = [
            np.median(np.asarray(bank[idx[0] : idx[-1] + 1]), axis=0)
            for idx in chunks
        ]

        median_psd = np.median(medians, axis=0)
        return median_psd

    @staticmethod
    def _to_float(x):
        if torch.is_tensor(x):
            return float(x.item())
        return float(x)

    def _write_recolour_bank_meta(self, save_dir, num_psds, num_freq, freqs, sample_rate):
        # The bank .bin is streamed to disk during estimate_raw_psds; here we
        # only emit the sidecar JSON the recolour module reads.
        meta_path = os.path.join(save_dir, f"raw_{self.detector}_psds.json")
        meta = {
            "detector": self.detector,
            "num_psds": num_psds,
            "num_freq_bins": num_freq,
            "dtype": "float32",
            "byte_order": "little",
            "layout": "row-major",
            "sample_rate": sample_rate,
            "delta_f": EstimatePSD._to_float(freqs[1] - freqs[0]),
            "freq_start": EstimatePSD._to_float(freqs[0]),
            "freq_end": EstimatePSD._to_float(freqs[-1]),
            "psd_method": self.psd_method.__class__.__name__,
            "apply_inverse_spectrum_truncation": self.apply_ist,
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "max_filter_len": self.max_filter_len,
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def _save_raw_psds_hdf5(self, bin_path, save_dir, num_psds, num_freq, freqs, sample_rate):
        # Archive the streamed bank as a gzip-compressed HDF5 dataset, copied in
        # chunks so the full bank is never held in memory at once.
        hdf5_path = os.path.join(save_dir, f"raw_{self.detector}_psds.h5")
        bank = np.memmap(
            bin_path, dtype=np.float32, mode="r", shape=(num_psds, num_freq)
        )
        step = 1000
        with h5py.File(hdf5_path, "w") as hf:
            dset = hf.create_dataset(
                "psds",
                shape=(num_psds, num_freq),
                dtype="float32",
                chunks=(min(step, num_psds), num_freq),
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            for start in range(0, num_psds, step):
                end = min(start + step, num_psds)
                dset[start:end] = np.asarray(bank[start:end])
            hf.create_dataset("freqs", data=freqs)
            hf.attrs["sample_rate"] = sample_rate
        del bank

    def _save_fiducial_psd(
        self,
        psd,
        freqs,
        blackout_idxs,
        fiducial_dir,
        sample_rate,
    ):
        # Fiducial PSDs saved in export directory
        if self.store_psds_as_hdf5:
            hdf5_path = os.path.join(fiducial_dir, f"fiducial_{self.detector}_psd.h5")

            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)

            with h5py.File(hdf5_path, "w") as hf:
                hf.create_dataset(
                    "psd",
                    data=psd,
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
                        "num_freq_bins": len(psd),
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

        elif self.store_psds_as_bin:
            bin_path = os.path.join(fiducial_dir, f"fiducial_{self.detector}_psd.bin")
            np.asarray(psd, dtype=np.float32).tofile(bin_path)

            meta = {
                "detector": self.detector,
                "num_freq_bins": len(psd),
                "dtype": "float32",
                "byte_order": "little",
                "sample_rate": sample_rate,
                "delta_f": EstimatePSD._to_float(freqs[1] - freqs[0]),
                "freq_start": EstimatePSD._to_float(freqs[0]),
                "freq_end": EstimatePSD._to_float(freqs[-1]),
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

    def estimate_segment_psds(self, *, noise_segments_file):
        """
        Compute Welch PSD for each noise segment in a bin file.

        Args:
            noise_segments_file: path to noise .bin file
            output_dir: directory to write PSD bin + metadata
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
        psd_bin_path = output_dir / f"data_{self.detector}_psds.bin"
        psd_meta_path = output_dir / f"data_{self.detector}_psds_segments.json"

        psd_meta = []
        psd_cursor = 0

        with open(psd_bin_path, "wb") as psd_fh:
            for seg in tqdm(seg_meta, desc="Computing PSDs per segment"):
                start = seg["sample_start_idx"]
                nsamp = seg["nsamples"]

                data = np.array(
                    mm[start : start + nsamp],
                    dtype=np.float32,
                    copy=True,
                )
                data /= DYN_RANGE_FAC

                ts = torch.from_numpy(data)

                psd = self.psd_method(ts).cpu().numpy()
                psd = np.sqrt(psd).astype(np.float32)
                freqs = self.psd_method.freqs
                delta_f = 1.0 / (self.psd_method.seg_len * self.psd_method.delta_t)

                # Apply inverse spectrum truncation
                if self.apply_ist:
                    raise NotImplementedError("Inverse spectrum truncation removed")
                    psd = torch.from_numpy(psd).to(torch.float64)
                    psd = inverse_spectrum_truncation_single(
                        psd=psd,
                        max_filter_len=self.max_filter_len,
                        low_frequency_cutoff=self.low_frequency_cutoff,
                        delta_f=delta_f,
                        trunc_method=self.trunc_method,
                    )
                    psd = psd.cpu().numpy()

                # Interpolate if requested
                if self.interpolate_psd:
                    psd, delta_f, freqs = EstimatePSD._interpolate(
                        psd=psd,
                        delta_f_psd=delta_f,
                        sample_length=self.training_sample_length,
                        sample_rate=self.data_cfg.sample_rate,
                    )

                # Spline smooth the PSD before saving
                # For PSD: 0.004 * len(freqs)
                # For ASD: 0.001 * len(freqs)
                if self.psd_smoothener is not None:
                    psd = self.psd_smoothener.smooth(
                        freqs, psd, smooth_factor=0.001 * len(freqs)
                    )

                # DO NOT do the following although its tempting
                # Kill all values below low frequency cutoff
                # psd[freqs < self.low_frequency_cutoff] = 1e30
                # This introduces long lasting ringing effects in TD
                # Instead we make a slow taper
                psd = self.taper(freqs, psd)

                nbytes = psd.nbytes
                psd_fh.write(psd.tobytes())

                psd_meta.append(
                    {
                        "noise_segment_index": seg["segment_index"],
                        "gps_start": seg["gps_start"],
                        "gps_end": seg["gps_end"],
                        "sample_rate": seg["sample_rate"],
                        "psd_len": psd.shape[0],
                        "byte_offset": psd_cursor,
                        "byte_length": nbytes,
                        "delta_f": delta_f,
                        "seg_len": self.psd_method.seg_len,
                        "seg_stride": self.psd_method.seg_stride,
                        "window": "hann",
                        "inverse_spectrum_truncation": 1 if self.apply_ist else 0,
                        "max_filter_len": self.max_filter_len,
                        "low_frequency_cutoff": self.low_frequency_cutoff,
                        "interpolation": 1 if self.interpolate_psd else 0,
                    }
                )

                psd_cursor += nbytes

        with open(psd_meta_path, "w") as f:
            json.dump(psd_meta, f, indent=2)
