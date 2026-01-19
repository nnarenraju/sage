#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : real_noise.py
Description   : Short description of the file

Created on 2026-01-19 14:23:24

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import h5py
import numpy as np

from pycbc import DYN_RANGE_FAC


class GenerateNoise:
    """
    Generate random noise slices from a monolithic HDF5 file.

    Selection logic:
    - Segments are chosen with probability proportional to their duration
    - A random contiguous slice of fixed length is returned
    """

    def __init__(self, h5_path: str, sample_rate: float):
        """
        Args:
            h5_path (str): Path to monolithic noise HDF5 file
            sample_rate (float): Sample rate of stored data (Hz)
        """
        self.h5_path = h5_path
        self.sample_rate = sample_rate

        self.hf = h5py.File(self.h5_path, "r")
        self.seg_grp = self.hf["segments"]

        # Sorted segment keys: ["0000", "0001", ...]
        self.segment_keys = sorted(self.seg_grp.keys())

        # Segment lengths in samples
        self.segment_lengths = np.array(
            [self.seg_grp[k].shape[0] for k in self.segment_keys],
            dtype=np.int64,
        )

        # Probability ∝ duration
        self.prob = self.segment_lengths / self.segment_lengths.sum()

    def _pick_segment(self):
        """Pick a segment index weighted by duration."""
        idx = np.random.choice(len(self.segment_keys), p=self.prob)
        key = self.segment_keys[idx]
        length = self.segment_lengths[idx]
        return self.seg_grp[key], length

    def _pick_start(self, seg_length, slice_length):
        """Pick a valid random start index inside a segment."""
        max_start = seg_length - slice_length
        if max_start <= 0:
            raise ValueError("Requested slice longer than segment.")
        return np.random.randint(0, max_start)

    def sample(self, duration: float):
        """
        Draw a random noise slice.

        Args:
            duration (float): Desired duration in seconds

        Returns:
            np.ndarray: Noise time series, shape (nsamples,)
        """
        nsamples = int(duration * self.sample_rate)

        while True:
            dset, seg_len = self._pick_segment()
            try:
                start = self._pick_start(seg_len, nsamples)
            except ValueError:
                continue

            noise = np.array(dset[start : start + nsamples], dtype=np.float64)
            noise /= DYN_RANGE_FAC

            if not np.any(np.isnan(noise)):
                return noise

    def close(self):
        self.hf.close()


class MultipleFileRandomNoiseSlice:
    """
    Same as RandomNoiseSlice but for multiple noise files with different durations in each
        1. Downloaded ~113 days of noise from O3b for H1 and L1
        2. PSDs shouldn't vary too drastically from O3a
        3. Each segment is at least 1 hour in length and stored in separate files

    """

    def __init__(
        self,
        noise_dirs=dict(
            H1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/H1",
            L1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/L1",
        ),
        lengths_dir=dict(
            H1="/home/nnarenraju/Research/sgwc-1/sage/notebooks/tmp/durs_H1_O3b_all_noise_deimos.npy",
            L1="/home/nnarenraju/Research/sgwc-1/sage/notebooks/tmp/durs_L1_O3b_all_noise_deimos.npy",
        ),
        debug_me=False,
        debug_dir="",
    ):

        # Noise data files
        self.detnames = list(noise_dirs.keys())
        self.sample_length = 0.0  # seconds
        self.noise_files = {}
        self.lengths = {}
        if os.path.isdir(noise_dirs[list(noise_dirs.keys())[0]]):
            for name in noise_dirs.keys():
                self.noise_files[name] = [
                    h5py.File(fname)
                    for fname in glob.glob(os.path.join(noise_dirs[name], "*.hdf"))
                ]
                # /home/nnarenraju/Research/sgwc-1/sage/notebooks/tmp/durs_{}_O3b_all_noise_deimos.npy
                self.lengths[name] = np.load(lengths_dir[name])
        else:
            for name in noise_dirs.keys():
                self.noise_files[name] = h5py.File(noise_dirs[name])
                self.lengths[name] = np.load(lengths_dir[name])

    def pick_noise_file(self, det):
        # Pick a noise file for each detector
        # Pick a random file to get the noise sample
        idx = np.random.choice(list(range(len(self.lengths[det]))))
        det_file, det_file_length = (
            self.noise_files[det]["data/chunk_{}".format(idx)],
            self.lengths[det][idx],
        )
        return det_file, det_file_length

    def _make_sample_start_time(self, seg_start_idx, seg_end_idx):
        # Make a sample start time that is uniformly distributed within segdur
        return int(np.random.uniform(low=seg_start_idx, high=seg_end_idx))

    def read_noise(self, hf, length, data_cfg, recolour_pad):
        # Get random noise segment
        seg_start_idx, seg_end_idx = (0, length - 1)
        seg_end_idx -= recolour_pad + self.sample_length * data_cfg.sample_rate
        # This start time will lie within a valid segment time interval
        sample_start_idx = self._make_sample_start_time(seg_start_idx, seg_end_idx)
        # Get the required portion of given segment
        sidx = sample_start_idx
        eidx = sample_start_idx + int(self.sample_length * data_cfg.sample_rate)
        eidx += recolour_pad
        # Get time series from segment and apply the dynamic range factor
        ts = np.array(hf[sidx:eidx]).astype(np.float64)
        ts /= DYN_RANGE_FAC
        return ts

    def apply(self, special, det_only=""):
        ## Get random noise sample for detector(s)
        if special["cfg"].transforms["noise"] != None:
            get_class = lambda clist, cname: [
                foo for foo in clist if foo.__class__.__name__ == cname
            ][0]
            recolour = get_class(
                special["cfg"].transforms["noise"].transforms, "Recolour"
            )
            recolour_flag = True if recolour != [] else False
        else:
            recolour_flag = False

        if special["training"]:
            if recolour_flag:
                recolour_pad = int(
                    special["data_cfg"].whiten_padding * special["data_cfg"].sample_rate
                )
            else:
                recolour_pad = 0
        else:
            recolour_pad = 0
        # Is the detector going to be augmented with extra noise?
        # is_augment = {"H1": np.random.rand() < 0.5, "L1": np.random.rand() < 0.5} -----------> Change
        is_augment = {
            self.detnames[0]: np.random.rand() < 1.0,
            self.detnames[1]: np.random.rand() < 1.0,
        }

        # Read the noise from the provided filenum
        noise_H1 = np.zeros(
            int(special["data_cfg"].sample_length_in_num + recolour_pad)
        )
        while True and is_augment[self.detnames[0]]:
            H1_file, H1_file_len = self.pick_noise_file(self.detnames[0])
            noise_H1 = self.read_noise(
                H1_file, H1_file_len, special["data_cfg"], recolour_pad
            )
            if not any(np.isnan(noise_H1)):
                break

        noise_L1 = np.zeros(
            int(special["data_cfg"].sample_length_in_num + recolour_pad)
        )
        while True and is_augment[self.detnames[1]]:
            L1_file, L1_file_len = self.pick_noise_file(self.detnames[1])
            noise_L1 = self.read_noise(
                L1_file, L1_file_len, special["data_cfg"], recolour_pad
            )
            if not any(np.isnan(noise_L1)):
                break

        # Augmented noise (Downsampled to 2048. Hz after downloading)
        noise = np.stack([noise_H1, noise_L1], axis=0)
        return noise
