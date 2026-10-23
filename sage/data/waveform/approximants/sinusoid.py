#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
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


class SinusoidGenerator:
    """
    Synthetic sinusoidal waveform generator for bias probing.

    Generates simple sine waves to test for spectral bias (varying frequency,
    fixed duration) and duration bias (varying duration, fixed frequency).
    Used to diagnose whether the network learns spurious correlations between
    signal properties and the detection score.

    Parameters
    ----------
    A : float
        Amplitude of the sinusoid.
    phi : float
        Initial phase (radians).
    inject_lower : float
        Lower bound for the random injection time within the segment (s).
    inject_upper : float
        Upper bound for the random injection time within the segment (s).
    spectral_bias : bool
        If ``True``, generate samples with varying frequency but fixed duration.
    fixed_duration : float
        Duration used when ``spectral_bias=True`` (s).
    lower_freq : float
        Lower frequency bound for spectral-bias sampling (Hz).
    upper_freq : float
        Upper frequency bound for spectral-bias sampling (Hz).
    duration_bias : bool
        If ``True``, generate samples with varying duration but fixed frequency.
    fixed_frequency : float
        Frequency used when ``duration_bias=True`` (Hz).
    lower_tau : float
        Lower duration bound for duration-bias sampling (s).
    upper_tau : float
        Upper duration bound for duration-bias sampling (s).
    no_whitening : bool
        Skip whitening-edge padding if ``True`` (default ``False``).
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
        """Return a pure sinusoid ``A * sin(2π f t + φ)`` sampled at times *t*."""
        return self.A * np.sin(2.0 * np.pi * f * t + self.phi)

    def get_time_shift(self, detectors):
        """Return the light-travel-time offset (seconds) between the two detectors."""
        # time shift signals based of detector choice
        ifo1, ifo2 = detectors
        dt = ifo1.light_travel_time_to_detector(ifo2)
        return dt

    def add_zero_padding(self, signal, start_time, sample_length, sample_rate):
        """Zero-pad *signal* to *sample_length* × *sample_rate* samples, placing
        the signal at *start_time*."""
        # if random duration less than sample_length, add zero padding
        left_pad = int(start_time * sample_rate)
        right_pad = int((sample_length * sample_rate - (left_pad + len(signal))))
        padded_signal = np.pad(
            signal, (left_pad, right_pad), "constant", constant_values=(0, 0)
        )

        return padded_signal

    def add_whiten_padding(self, signal, special):
        """Append symmetric whitening-corruption padding to *signal*."""
        # whiten padding added separately for ease of understanding
        padding = special["data_cfg"].whiten_padding
        left_pad = right_pad = int((padding / 2.0) * special["data_cfg"].sample_rate)
        padded_signal = np.pad(
            signal, (left_pad, right_pad), "constant", constant_values=(0, 0)
        )
        return padded_signal

    def testing_spectral_bias(self, special):
        """
        Generate two-detector sinusoid injections at random frequencies (fixed duration).

        Used to probe whether the model exhibits a spectral bias — i.e. favours
        certain frequency ranges.

        Returns
        -------
        numpy.ndarray, shape ``(2, N)``
            Per-detector time-series.
        """
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
        """
        Generate two-detector sinusoid injections at random durations (fixed frequency).

        Used to probe whether the model exhibits a duration bias — i.e. favours
        signals of certain in-band durations.

        Returns
        -------
        numpy.ndarray, shape ``(2, N)``
            Per-detector time-series.
        """
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
        """
        Generate sinusoidal test signals for bias probing.

        Dispatches to :meth:`testing_spectral_bias` or
        :meth:`testing_duration_bias` depending on the instance flags.

        Parameters
        ----------
        params : dict
            Unused; present for API compatibility with other waveform generators.
        special : dict
            Context dict containing ``'dets'`` (detector pair) and ``'data_cfg'``.

        Returns
        -------
        numpy.ndarray, shape ``(2, N)``
            Stacked per-detector time-series.
        """
        ## Generate sin waves for testing biased learning
        # Generate data based on required bias
        if self.spectral_bias:
            signals = self.testing_spectral_bias(special)
        elif self.duration_bias:
            signals = self.testing_duration_bias(special)
        return signals
