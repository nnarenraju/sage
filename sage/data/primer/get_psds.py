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
from tqdm import tqdm

# LOCAL
from sage.primer.blackout import NoBlackout
from sage.dsp.welch import WelchPSD


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
        store_raw_psds: bool = False,
    ):
        self.detector = detector
        self.num_samples = int(num_samples)
        self.psd_method = psd_method
        self.blackout_policy = blackout_policy or NoBlackout()
        self.store_raw_psds = store_raw_psds

    def __call__(self, *, noise_source, **kwargs):
        """
        Run PSD estimation.

        Expected kwargs injected by CodeflowManager:
            - noise_source
            - cfg
            - data_cfg
            - rng
        """

        # Pull required runtime context
        sample_rate = kwargs["data_cfg"].sample_rate
        duration = kwargs["data_cfg"].sample_length

        # Save directories for PSDs and Fiducial PSDs
        export_dir = os.path.join(kwargs["cfg"].export_dir, "fiducial_psds")
        data_dir = os.path.join(kwargs["data_cfg"].data_dir, "psds", self.detector)

        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        psds = []

        for _ in tqdm(
            range(self.num_samples),
            desc=f"Estimating PSDs for {self.detector}",
        ):
            # Sample noise sample given duration
            noise = noise_source(duration)

            # Compute PSD using the Welch method
            freqs, pxx = self.psd_method(noise)

            # To save each PSD if requested
            psds.append(pxx)

        # Put all PSDs together into one unit
        psds = np.stack(psds, axis=0)

        # Store each raw PSD for recolouring module
        if self.store_raw_psds:
            self._save_raw_psds(psds, freqs, data_dir, sample_rate)

        # Compute median PSD, blackout difficult regions
        median_psd = self._aggregate_psds(psds)
        fiducial_psd, blackout_idxs = self.blackout_policy.apply(median_psd, psds)

        # Saving fiducial PSD in export_dir of run
        self._save_fiducial_psd(
            fiducial_psd,
            freqs,
            blackout_idxs,
            export_dir,
            sample_rate,
        )

        return fiducial_psd, freqs

    def _aggregate_psds(self, psds):
        # Median of medians is memory efficient
        chunks = np.array_split(psds, max(1, psds.shape[0] // 10_000))
        medians = [np.median(chunk, axis=0) for chunk in chunks]

        median_psd = np.median(medians, axis=0)
        return median_psd

    def _save_raw_psds(self, psds, freqs, data_dir, sample_rate):
        # Save all raw PSDs into one file
        # Saved inside individual detector directories
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
        # Fiducial PSDs saved in export directory
        path = os.path.join(export_dir, f"fiducial_{self.detector}_psd.h5")

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
                    "blackout_policy": self.blackout_policy.__class__.__name__,
                    "num_samples": self.num_samples,
                    "sample_rate": sample_rate,
                }
            )
