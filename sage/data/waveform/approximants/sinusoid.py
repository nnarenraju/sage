#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sinusoid test signals for probing spectral and duration bias of a
detection network.

Filename      : sinusoid.py
Description   : Short description of the file

Created on 2026-01-19 16:06:34

__author__      = Narenraju Nagarajan
__copyright__   = Copyright 2026, Sage
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = Narenraju Nagarajan
__email__       = N/A
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

import numpy as np


class SinusoidGenerator:
    """
    Sinusoid generator for spectral- and duration-bias tests.

    Spectral bias: random frequency at a fixed duration. Duration bias:
    random duration at a fixed frequency.

    Parameters
    ----------
    A : float
        Amplitude of the sinusoid.
    phi : float
        Initial phase, in rad.
    inject_lower : float
        Lower bound of the random injection time in the segment, in s.
    inject_upper : float
        Upper bound of the random injection time in the segment, in s.
    spectral_bias : bool
        Draw a random frequency at a fixed duration.
    fixed_duration : float
        Duration for spectral-bias samples, in s.
    lower_freq : float
        Lower bound of the spectral-bias frequency, in Hz.
    upper_freq : float
        Upper bound of the spectral-bias frequency, in Hz.
    duration_bias : bool
        Draw a random duration at a fixed frequency.
    fixed_frequency : float
        Frequency for duration-bias samples, in Hz.
    lower_tau : float
        Lower bound of the duration-bias duration, in s.
    upper_tau : float
        Upper bound of the duration-bias duration, in s.
    no_whitening : bool
        Skip the whitening padding.
    """

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
        # Sinusoid parameters
        self.A = A
        self.phi = phi
        self.inject_lower = inject_lower
        self.inject_upper = inject_upper
        # Spectral bias (same duration, different frequencies)
        self.spectral_bias = spectral_bias
        self.fixed_duration = fixed_duration
        self.lower_freq = lower_freq
        self.upper_freq = upper_freq
        # Duration bias (same frequency, different durations)
        self.duration_bias = duration_bias
        self.fixed_frequency = fixed_frequency
        self.lower_tau = lower_tau
        self.upper_tau = upper_tau
        self.no_whitening = no_whitening

    def generate(self, f, t):
        """
        Sinusoid A sin(2 pi f t + phi) at times t.

        Parameters
        ----------
        f : float
            Frequency in Hz.
        t : numpy.ndarray
            Sample times in s.

        Returns
        -------
        numpy.ndarray
            Sinusoid at each time in t.
        """
        return self.A * np.sin(2.0 * np.pi * f * t + self.phi)

    def get_time_shift(self, detectors):
        """
        Light travel time between two detectors.

        Parameters
        ----------
        detectors : sequence
            Two detectors with light_travel_time_to_detector.

        Returns
        -------
        float
            Light travel time between the two detectors, in s.
        """
        ifo1, ifo2 = detectors
        dt = ifo1.light_travel_time_to_detector(ifo2)
        return dt

    def add_zero_padding(self, signal, start_time, sample_length, sample_rate):
        """
        Zero-pad a signal to the sample length, starting at start_time.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal samples.
        start_time : float
            Start of the signal in the padded output, in s.
        sample_length : float
            Length of the padded output, in s.
        sample_rate : float
            Sample rate in Hz.

        Returns
        -------
        numpy.ndarray
            Zero-padded signal of sample_length * sample_rate samples.
        """
        left_pad = int(start_time * sample_rate)
        right_pad = int((sample_length * sample_rate - (left_pad + len(signal))))
        padded_signal = np.pad(
            signal, (left_pad, right_pad), "constant", constant_values=(0, 0)
        )

        return padded_signal

    def add_whiten_padding(self, signal, special):
        """
        Zero-pad both ends of a signal by half the whitening padding.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal samples.
        special : dict
            Context with 'data_cfg' (whiten_padding, sample_rate).

        Returns
        -------
        numpy.ndarray
            Padded signal.
        """
        padding = special["data_cfg"].whiten_padding
        left_pad = right_pad = int((padding / 2.0) * special["data_cfg"].sample_rate)
        padded_signal = np.pad(
            signal, (left_pad, right_pad), "constant", constant_values=(0, 0)
        )
        return padded_signal

    def testing_spectral_bias(self, special):
        """
        Two-detector sinusoid at a random frequency, fixed duration.

        Parameters
        ----------
        special : dict
            Context with 'dets' (detector pair) and 'data_cfg'.

        Returns
        -------
        numpy.ndarray, shape (2, N)
            Time series of each detector; the second detector is offset
            by the light travel time.
        """
        detectors = special["dets"]
        sample_length = special["data_cfg"].signal_length  # seconds
        sample_rate = special["data_cfg"].sample_rate  # Hz
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
            signal, start_time + dt, sample_length, sample_rate
        )
        # Add whiten padding separately
        if not self.no_whitening:
            signal_det1 = self.add_whiten_padding(signal_det1, special)
            signal_det2 = self.add_whiten_padding(signal_det2, special)
        return np.stack((signal_det1, signal_det2), axis=0)

    def testing_duration_bias(self, special):
        """
        Two-detector sinusoid at a random duration, fixed frequency.

        Parameters
        ----------
        special : dict
            Context with 'dets' (detector pair) and 'data_cfg'.

        Returns
        -------
        numpy.ndarray, shape (2, N)
            Time series of each detector; the second detector is offset
            by the light travel time.
        """
        detectors = special["dets"]
        sample_length = special["data_cfg"].signal_length  # seconds
        sample_rate = special["data_cfg"].sample_rate  # Hz
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
        """
        Sinusoid test signal of the bias selected at construction.

        Parameters
        ----------
        params : dict
            Not used.
        special : dict
            Context with 'dets' (detector pair) and 'data_cfg'.

        Returns
        -------
        numpy.ndarray, shape (2, N)
            Time series of each detector.
        """
        if self.spectral_bias:
            signals = self.testing_spectral_bias(special)
        elif self.duration_bias:
            signals = self.testing_duration_bias(special)
        return signals
