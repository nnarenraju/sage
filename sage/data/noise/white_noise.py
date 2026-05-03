#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : white_noise.py
Description   : Short description of the file

Created on 2026-01-19 16:18:49

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


class WhiteNoiseGenerator:
    """
    Generate independent white Gaussian noise for each detector.

    Produces zero-mean, unit-variance Gaussian noise with independent seeds
    per detector.  Primarily used for controlled testing and as a substitute
    for real noise during pipeline development or unit tests.
    """

    def generate(self, sample_length_in_num, seed=0):
        """
        Draw a single white Gaussian noise realisation.

        Parameters
        ----------
        sample_length_in_num : int
            Number of samples to generate.
        seed : int
            NumPy random seed for reproducibility.

        Returns
        -------
        numpy.ndarray, shape ``(sample_length_in_num,)``
            Zero-mean, unit-variance Gaussian noise.
        """
        np.random.seed(seed)
        # 0 mean, 1 std
        return np.random.normal(0, 1, size=sample_length_in_num)

    def apply(self, special, det_only=""):
        """
        Generate dual-detector white noise for a single sample.

        Parameters
        ----------
        special : dict
            Must contain ``"sample_seed"`` (int) and ``"data_cfg"`` with
            ``signal_length`` (s) and ``sample_rate`` (Hz) attributes.
        det_only : str
            Unused; kept for API compatibility.

        Returns
        -------
        numpy.ndarray, shape ``(2, N)``
            Stacked H1/L1 white noise arrays.
        """
        # Generate white Gaussian noise using random seeds
        rs = np.random.RandomState(seed=special["sample_seed"])
        seeds = list(rs.randint(0, 2**32, 2))  # one for each detector
        # Get sample length in num
        sample_length_in_s = special["data_cfg"].signal_length  # in seconds
        sample_rate = special["data_cfg"].sample_rate  # in samples/second
        sample_length_in_num = int(sample_length_in_s * sample_rate)
        # Generate noise for each detector
        H1_noise = self.generate(sample_length_in_num, seeds[0])
        L1_noise = self.generate(sample_length_in_num, seeds[1])
        # Return noise to dataset object
        noise = np.stack([H1_noise, L1_noise], axis=0)
        return noise
