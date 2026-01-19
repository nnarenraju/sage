#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename      : sinusoid.py
Description   : Short description of the file

Created on 2026-01-19 16:06:34

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


class SinusoidGenerator:
    ## Used to create sinusoid with different parameters to test biases
    ## Bias due to waveform frequency comes under spectral bias
    ## Bias due to signal duration comes under lack of proper inductive bias
    def __init__(
        self,
        A,
        phi,
        inject_lower=2.0,
        inject_upper=3.0,
        spectral_bias=False,
        fixed_duration=5.0,
        lower_freq=20.0,
        upper_freq=1024.0,
        duration_bias=False,
        fixed_frequency=100.0,
        lower_tau=0.1,
        upper_tau=5.0,
        no_whitening=False,
    ):
        # Sinusoidal wave parameters in general form
        self.A = A
        self.phi = phi
        self.inject_lower = inject_lower
        self.inject_upper = inject_upper
        # Spectral Bias (same duration, different frequencies)
        self.spectral_bias = spectral_bias
        self.fixed_duration = fixed_duration
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        # Duration bias (same frequency, different durations)
        self.duration_bias = duration_bias
        self.fixed_frequency = fixed_frequency
        self.lower_tau = lower_tau
        self.upper_tau = upper_tau
        # Other options
        self.no_whitening = no_whitening

    def generate(self, f, t):
        return self.A * np.sin(2.0 * np.pi * f * t + self.phi)

    def get_time_shift(self, detectors):
        # time shift signals based of detector choice
        ifo1, ifo2 = detectors
        dt = ifo1.light_travel_time_to_detector(ifo2)
        return dt

    def add_zero_padding(self, signal, start_time, sample_length, sample_rate):
        # if random duration less than sample_length, add zero padding
        left_pad = int(start_time * sample_rate)
        right_pad = int((sample_length * sample_rate - (left_pad + len(signal))))
        padded_signal = np.pad(
            signal, (left_pad, right_pad), "constant", constant_values=(0, 0)
        )

        return padded_signal

    def add_whiten_padding(self, signal, special):
        # whiten padding added separately for ease of understanding
        padding = special["data_cfg"].whiten_padding
        left_pad = right_pad = int((padding / 2.0) * special["data_cfg"].sample_rate)
        padded_signal = np.pad(
            signal, (left_pad, right_pad), "constant", constant_values=(0, 0)
        )
        return padded_signal

    def testing_spectral_bias(self, special):
        ## Generating sin waves with different frequencies but same duration
        # Params
        detectors = special["dets"]
        sample_length = special["data_cfg"].signal_length  # seconds
        sample_rate = special["data_cfg"].sample_rate  # Hz
        # Simulating bias
        random_freq = np.random.uniform(low=self.lower_freq, high=self.upper_freq)
        tseries = np.linspace(
            0.0, self.fixed_duration, int(self.fixed_duration * sample_rate)
        )
        # Get time shift between detectors
        dt = self.get_time_shift(detectors)
        signal = self.generate(random_freq, tseries)
        start_time = np.random.uniform(self.inject_lower, self.inject_upper)
        signal_det1 = self.add_zero_padding(
            signal, start_time, sample_length, sample_rate
        )
        # Add dt to start time for detector offset
        signal_det2 = self.add_zero_padding(
            signal, start_time, sample_length, sample_rate
        )
        # Add whiten padding separately
        if not self.no_whitening:
            signal_det1 = self.add_whiten_padding(signal_det1, special)
            signal_det2 = self.add_whiten_padding(signal_det2, special)
        return np.stack((signal_det1, signal_det2), axis=0)

    def testing_duration_bias(self, special):
        ## Generating sin waves with different duration but same frequency
        # Params
        detectors = special["dets"]
        sample_length = special["data_cfg"].signal_length  # seconds
        sample_rate = special["data_cfg"].sample_rate  # Hz
        # Simulating bias
        random_dur = np.random.uniform(low=self.lower_tau, high=self.upper_tau)
        tseries = np.linspace(0.0, random_dur, int(random_dur * sample_rate))
        # Get time shift between detectors
        dt = self.get_time_shift(detectors)
        signal = self.generate(self.fixed_frequency, tseries)
        start_time = np.random.uniform(self.inject_lower, self.inject_upper)
        signal_det1 = self.add_zero_padding(
            signal, start_time, sample_length, sample_rate
        )
        signal_det2 = self.add_zero_padding(
            signal, start_time + dt, sample_length, sample_rate
        )
        # Add whiten padding separately
        if not self.no_whitening:
            signal_det1 = self.add_whiten_padding(signal_det1, special)
            signal_det2 = self.add_whiten_padding(signal_det2, special)
        return np.stack((signal_det1, signal_det2), axis=0)

    def apply(self, params: dict, special: dict):
        ## Generate sin waves for testing biased learning
        # Generate data based on required bias
        if self.spectral_bias:
            signals = self.testing_spectral_bias(special)
        elif self.duration_bias:
            signals = self.testing_duration_bias(special)
        return signals
