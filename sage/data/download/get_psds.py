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
import numpy as np
from scipy import signal as ss
from tqdm import tqdm


class EstimatePSD:
    """
    Estimate a fiducial PSD by sampling noise from the active noise pipeline.
    """

    def __init__(
        self,
        *,
        detector: str,
        num_samples: int = 200_000,
        nperseg_seconds: float = 4.0,
        welch_average: str = "median",
        blackout_max_ratio: float = 2.0,
        store_raw_psds: bool = False,
    ):
        self.detector = detector
        self.num_samples = int(num_samples)
        self.blackout_max_ratio = blackout_max_ratio
        self.store_raw_psds = store_raw_psds

        self.nperseg_seconds = nperseg_seconds
        self.welch_average = welch_average

    def __call__(self, *, noise_source, cfg=None, data_cfg=None, rng=None, **_):
        """
        Run PSD estimation.

        Expected kwargs injected by CodeflowManager:
            - noise_source
            - cfg
            - data_cfg
            - rng
        """

        # Pull required runtime context
        sample_rate = data_cfg.sample_rate
        duration = data_cfg.sample_length

        export_dir = os.path.join(cfg.export_dir, "psd", self.detector)
        data_dir = os.path.join(data_cfg.data_dir, "psd", self.detector)

        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        nperseg = int(self.nperseg_seconds * sample_rate)

        psds = []
        freqs = None

        for _ in tqdm(
            range(self.num_samples),
            desc=f"Estimating PSD ({self.detector})",
        ):
            noise = noise_source.run(duration)

            if noise.ndim == 2:
                noise = noise[0]

            f, pxx = ss.welch(
                noise,
                fs=sample_rate,
                nperseg=nperseg,
                average=self.welch_average,
            )

            if freqs is None:
                freqs = f

            psds.append(pxx)

        psds = np.stack(psds, axis=0)

        if self.store_raw_psds:
            self._save_raw_psds(psds, freqs, data_dir, sample_rate)

        median_psd, max_psd = self._aggregate_psds(psds)
        blacked_psd, blackout_idxs = self._apply_blackout(median_psd, max_psd)

        self._save_fiducial_psd(
            blacked_psd,
            freqs,
            blackout_idxs,
            export_dir,
            sample_rate,
        )

        return blacked_psd, freqs

    def _aggregate_psds(self, psds):
        chunks = np.array_split(psds, max(1, psds.shape[0] // 10_000))
        medians = [np.median(chunk, axis=0) for chunk in chunks]

        median_psd = np.median(medians, axis=0)
        max_psd = np.maximum.reduce(medians)

        return median_psd, max_psd

    def _apply_blackout(self, median_psd, max_psd):
        ratio = max_psd / median_psd
        blackout_idxs = np.where(ratio > self.blackout_max_ratio)[0]

        blacked_psd = median_psd.copy()
        blacked_psd[blackout_idxs] = 1e12

        frac = len(blackout_idxs) / len(ratio)
        print(f"[{self.detector}] " f"Blacked out {frac*100:.2f}% of frequency bins")

        return blacked_psd, blackout_idxs

    def _save_raw_psds(self, psds, freqs, data_dir, sample_rate):
        path = os.path.join(data_dir, "raw_psds.h5")

        with h5py.File(path, "w") as hf:
            hf.create_dataset(
                "psds",
                data=psds,
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            hf.create_dataset("freqs", data=freqs)
            hf.attrs["sample_rate"] = sample_rate

    def _save_fiducial_psd(
        self,
        psd,
        freqs,
        blackout_idxs,
        export_dir,
        sample_rate,
    ):
        path = os.path.join(export_dir, "fiducial_psd.h5")

        if os.path.exists(path):
            os.remove(path)

        with h5py.File(path, "w") as hf:
            hf.create_dataset(
                "psd",
                data=psd,
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            hf.create_dataset("freqs", data=freqs)
            hf.create_dataset("blackout_indices", data=blackout_idxs)

            hf.attrs.update(
                {
                    "detector": self.detector,
                    "delta_f": freqs[1] - freqs[0],
                    "blackout_max_ratio": self.blackout_max_ratio,
                    "num_samples": self.num_samples,
                    "sample_rate": sample_rate,
                }
            )
