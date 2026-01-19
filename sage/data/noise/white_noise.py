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
    """Generate white Gaussian noise for Sage training"""

    def generate(self, sample_length_in_num, seed=0):
        """Generate data with a white Gaussian (normal) distribution"""
        np.random.seed(seed)
        # 0 mean, 1 std
        return np.random.normal(0, 1, size=sample_length_in_num)

    def apply(self, special, det_only=""):
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
