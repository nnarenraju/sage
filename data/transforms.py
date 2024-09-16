# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Jan 27 19:54:16 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

# BUILT-IN
import os
import csv
import glob
import uuid
import h5py
import time
import random
import warnings
import numpy as np

from scipy.signal import butter, sosfiltfilt, resample, get_window
from scipy.signal import welch as scipy_welch
from scipy.signal.windows import tukey
from scipy.stats import beta
from scipy.stats import halfnorm
from numpy.random import RandomState
from scipy.signal import decimate

# LOCAL
from data.multirate_sampling import multirate_sampling
from data.snr_calculation import get_network_snr
from data.mlmdc_noise_generator import NoiseGenerator

# PyCBC
import pycbc
from pycbc import DYN_RANGE_FAC
from pycbc.detector import Detector
from pycbc.filter import highpass as pycbc_highpass
from pycbc.psd import inverse_spectrum_truncation, welch, interpolate
from pycbc.types import TimeSeries, FrequencySeries, load_frequencyseries, complex_same_precision_as

# LALSimulation Packages
import lalsimulation as lalsim

# Using segments to read O3a noise
import requests
import ligo.segments

# Plotting
import matplotlib.pyplot as plt

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# This constant need to be constant to be able to recover identical results.
BLOCK_SAMPLES = 1638400


""" UTILS """

def coin(pof1=0.5):
    return 1 if np.random.random() < pof1 else 0


""" WRAPPERS """

class Unify:
    def __init__(self, transforms: dict):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, special: dict, key=None):
        transforms = {}
        for transform in self.transforms[key]:
            name = transform.__class__.__name__
            y = transform(y, special)
            transforms[name] = y
        transforms['sample'] = y
        return transforms


class UnifySignal:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, params: dict, special: dict, debug=None, return_metadata=False):
        if return_metadata:
            get_vars = lambda cls: {key:val for key, val in cls.__dict__.items() if not key.startswith('__') and not callable(key)}
            return {foo.__class__.__name__: get_vars(foo) for foo in self.transforms}
        for transform in self.transforms:
            y, params, special = transform(y, params, special, debug)
        return (y, params, special)


class UnifySignalGen:
    def __init__(self, generations: list):
        self.generations = generations
    
    def __call__(self, params: dict, special: dict):
        for generation in self.generations:
            y = generation.apply(params, special)
        return y


class UnifyNoise:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, debug=None):
        for transform in self.transforms:
            y = transform(y, debug)
        return y


class UnifyNoiseGen:
    def __init__(self, generations, aux=None, paux=0.0, debug_me=False, debug_dir=""):
        self.generations = generations
        self.aux = aux
        # Probability of selecting primary generation method or aux
        self.paux = paux
        # Debugging
        self.debug_me = debug_me
        if debug_me:
            self.debug_dir = debug_dir
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
            save_txt = os.path.join(debug_dir, 'unify_noise_gen.txt')
            self.tmp_debug = open(save_txt, "a")
    
    def debug_noise_generator(self, data, labels):
        # Debugging plotter for gwspy
        fig, axs = plt.subplots(len(labels), 1, figsize=(9.0, 9.0*len(labels)), squeeze=False)
        fig.suptitle('Debugging Noise Generator')
        for n, (d, l) in enumerate(zip(data, labels)):
            # Subplot
            axs[n][0].plot(d, label=l)
            axs[n][0].grid()
            axs[n][0].legend()
        # Other
        filename = 'noise_generator_backend_{}.png'.format(uuid.uuid4().hex)
        save = os.path.join(self.debug_dir, filename)
        plt.savefig(save)
        plt.close()
    
    def __call__(self, special: dict):
        det_only = "none"
        if special['training']:
            do_aux = np.random.rand() < self.paux
            if do_aux and self.aux != None:
                # Has probability to set one or more dets to zeros
                y = self.aux.apply(special)
                # If both detectors are set to zeros
                if not np.any(y[0]) and not np.any(y[1]):
                    y = self.generations['training'].apply(special)
                # If only one detector is set to zeros
                elif not np.any(y[0]) or not np.any(y[1]):
                    det_only = 'H1' if not np.any(y[0]) else None
                    det_only = 'L1' if not np.any(y[1]) else det_only
                    # Get noise for one det only and set noise
                    det_noise = self.generations['training'].apply(special, det_only)
                    y[0 if det_only=='H1' else 1] = det_noise
            else:
                y = self.generations['training'].apply(special)
        else:
            y = self.generations['validation'].apply(special)
        
        # Debugging
        if self.debug_me:
            data = [y[0], y[1]]
            labels = ['H1 final noise sample', 'L1 final noise sample']
            self.debug_noise_generator(data, labels)
            foo = "{}".format(det_only)
            self.tmp_debug.write(foo)

        return y
    

class TransformWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray, special: dict):
        if self.always_apply:
            return self.apply(y, special)
        else:
            pass


class TransformWrapperPerChannel(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply=always_apply)
    
    def __call__(self, y: np.ndarray, special: dict):
        channels = y.shape[0]
        # Store transformed array
        augmented = []
        for channel in range(channels):
            if self.always_apply:
                augmented.append(self.apply(y[channel], channel, special))
            else:
                pass
        return np.stack(augmented, axis=0)


class SignalWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray, params: dict, special: dict, debug=None):
        if self.always_apply:
            return self.apply(y, params, special, debug)
        else:
            pass


class NoiseWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray, debug=''):
        if self.always_apply:
            return self.apply(y, debug)


####################################################################################################
#                             Transforms & their Functionality
# [0] Buffer - Absolutely nothing, say it again y'all
# [1] Normalise - Normalisation of each sample wrt entire dataset (0.0008s)
# [2] BandPass - Butter bandpass filter. Uses sosfiltfilt function for stability. (0.0035s) 
# [3] Whitening - PyCBC whitening function. PSD input required to whiten. (0.09s)
# [4] Multi-rate sampling - Sampling with multiple rates based on GW freq. (0.02s or less)
# [5] AugmentPolSky - Augmenting on polarisation and sky position. (0.0325s)
# [6] CyclicShift - Time shift noise samples. (5e-5s or less)
# [7] AugmentDistance - Augmenting on GW distance. (0.01s or less)
# 
####################################################################################################

class Buffer(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, special: dict):
        return y


class BufferPerChannel(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, channel: int, special: dict):
        return y


class Normalise(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, factors=[1.0, 1.0], ignore_factors=False):
        super().__init__(always_apply)
        assert len(factors) == 2
        self.factors = factors
        self.ignore_factors = ignore_factors

    def apply(self, y: np.ndarray, channel: int, special: dict):
        if not self.ignore_factors:
            norm = y / self.factors[channel]
        else:
            norm = (y - np.min(y)) / (np.max(y) - np.min(y)) # varies from 0 to 1
            norm = norm - np.mean(norm) # centering at 0
        return norm


class Resample(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, factor=2.):
        super().__init__(always_apply)
        self.factor = factor

    def apply(self, y: np.ndarray, channel: int, special: dict):
        new_len = int(len(y) * self.factor)
        resampled = resample(y, new_len)
        return resampled


class BandPass(TransformWrapper):
    def __init__(self, always_apply=True, lower=16, upper=512, fs=2048, order=5):
        super().__init__(always_apply)
        self.lower = lower
        self.upper = upper
        self.fs = fs
        self.order = order
    
    def butter_bandpass(self):
        nyq = 0.5 * self.fs
        low = self.lower / nyq
        high = self.upper / nyq
        sos = butter(self.order, [low, high], analog=False, btype='bandpass', output='sos')
        return sos

    def butter_bandpass_filter(self, data):
        sos = self.butter_bandpass()
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data
    
    def apply(self, y: np.ndarray, special: dict):
        return self.butter_bandpass_filter(y)


class HighPass(TransformWrapper):
    def __init__(self, always_apply=True, lower=16, fs=2048, order=5):
        super().__init__(always_apply)
        self.lower = lower
        self.fs = fs
        self.order = order
    
    def butter_highpass(self):
        nyq = 0.5 * self.fs
        low = self.lower / nyq
        sos = butter(self.order, low, analog=False, btype='highpass', output='sos')
        return sos

    def butter_highpass_filter(self, data):
        sos = self.butter_highpass()
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data
    
    def apply(self, y: np.ndarray, special: dict):
        # Parallelise HighPass filter
        return self.butter_highpass_filter(y)


class Crop(TransformWrapperPerChannel):
    # Crop the signal if required
    # If not whitening, we use this to emulate the remove_corrupted = True option
    def __init__(self, always_apply=True, croplen=0.0, emulate_rmcorrupt=False, double_sided=True, side='left'):
        super().__init__(always_apply)
        self.emulate_rmcorrupt = emulate_rmcorrupt
        self.croplen = croplen
        self.double_sided = double_sided
        self.side = side

    def get_cropped(self, y, data_cfg):
        if self.emulate_rmcorrupt:
            self.croplen = int(round(data_cfg.whiten_padding * data_cfg.sample_rate))
        if self.double_sided:
            cropped = y[int(self.croplen/2):int(len(y)-self.croplen/2)]
        else:
            if self.side == 'left':
                cropped = y[int(self.croplen):]
            elif self.side == 'right':
                cropped = y[:int(len(y)-self.croplen)]
            else:
                raise ValueError('Crop Transform: side not recognised!')
        return cropped
    
    def apply(self, y: np.ndarray, channel: int, special: dict):
        return self.get_cropped(y, special['data_cfg'])


class Whiten(TransformWrapperPerChannel):
    # PSDs can be different between the channels, so we use perChannel method
    def __init__(self, always_apply=True, trunc_method='hann', remove_corrupted=True, estimated=False):
        super().__init__(always_apply)
        self.trunc_method = trunc_method
        self.remove_corrupted = remove_corrupted
        self.estimated = estimated
        self.median_psd = []
    
    def estimate_psd(self, data_cfg, delta_f, max_filter_len):
        ### Estimate the PSD
        delta_t = 1.0/2048.
        seg_len = int(0.5 / delta_t)
        seg_stride = int(seg_len / 2)
        pure_noise = TimeSeries(pure_noise, delta_t=1./data_cfg.sample_rate)
        psd = welch(pure_noise, seg_len=seg_len, seg_stride=seg_stride)
        psd = interpolate(psd, delta_f)
        psd = inverse_spectrum_truncation(psd,
                                        max_filter_len=max_filter_len,
                                        low_frequency_cutoff=data_cfg.signal_low_freq_cutoff,
                                        trunc_method=self.trunc_method)
        return psd

        
    def whiten(self, signal, psd, data_cfg):
        """ Return a whitened time series """
        # Convert signal to Timeseries object
        signal = TimeSeries(signal, delta_t=1./data_cfg.sample_rate)
        # Filter length for inverse spectrum truncation
        max_filter_len = int(round(data_cfg.whiten_padding * data_cfg.sample_rate))
        
        ## Manipulate PSD for usage in whitening
        ## Interpolation is probably not required as the psds are created based on signal len anyway
        # Calculating delta_f of signal and providing that to the PSD interpolation method
        delta_f = data_cfg.delta_f
        # Interpolate the PSD to the required delta_f
        # NOTE: This may or may not be required (enable if there is a discrepancy in delta_f)
        # Possible bug: It is possible that the sample lengths are not consistent in Dataloader
        psd = interpolate(psd, delta_f)
        
        ## Whitening
        # Whiten the data by the asd
        if self.estimated:
            raise NotImplementedError('PSD Estimation method not working at the moment!')
        
        # Interpolate and smooth to the desired corruption length
        psd = inverse_spectrum_truncation(psd,
                                        max_filter_len=max_filter_len,
                                        low_frequency_cutoff=data_cfg.signal_low_freq_cutoff,
                                        trunc_method=self.trunc_method)
        
        # NOTE: Factor of dt not taken into account. Since layernorm takes care of standardisation,
        # we don't necessarily need to include this. After decorrelation, the diagonal matrix 
        # values will not be 1 but some other value dependant on the signal input.
        white = (signal.to_frequencyseries(delta_f=delta_f) / psd**0.5).to_timeseries()
        
        if self.remove_corrupted:
            white = white[int(max_filter_len/2):int(len(white)-max_filter_len/2)]

        return white
    
    def apply(self, y: np.ndarray, channel: int, special: dict):
        # Whitening using approximate PSD
        return self.whiten(y, special['psds'][channel], special['data_cfg'])


class MultirateSampling(TransformWrapperPerChannel):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, channel: int, special: dict):
        # Call multi-rate sampling module for usage
        # This module is kept separate since further experimentation might be required
        return multirate_sampling(y, special['data_cfg'])


class MonorateSampling(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, sampling_rate=2048.):
        super().__init__(always_apply)
        self.new_sample_rate = sampling_rate

    def decimate(self, signal, data_cfg):
        decimation_factor = int(round(data_cfg.sample_rate/self.new_sample_rate))
        # Decimation factor > 13 requires multiple sequential decimation
        # See multirate_sampling.py for relevant code
        assert decimation_factor <= 13, 'Decimation factor > 13! Not supported.'
        return decimate(signal, decimation_factor)
    
    def apply(self, y: np.ndarray, channel: int, special: dict):
        # Call multi-rate sampling module for usage
        # This module is kept separate since further experimentation might be required
        return self.decimate(y, special['data_cfg'])


""" Waveform Generation """


class SinusoidGenerator():
    ## Used to create sinusoid with different parameters to test biases
    ## Bias due to waveform frequency comes under spectral bias
    ## Bias due to signal duration comes under lack of proper inductive bias
    def __init__(self, 
                 A, 
                 phi, 
                 inject_lower = 2.0,
                 inject_upper = 3.0,
                 spectral_bias = False,
                 fixed_duration = 5.0,
                 lower_freq = 20.0,
                 upper_freq = 1024.0, 
                 duration_bias = False,
                 fixed_frequency = 100.0,
                 lower_tau = 0.1,
                 upper_tau = 5.0,
                 no_whitening = False,
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
        return self.A * np.sin(2.*np.pi*f*t + self.phi)

    def get_time_shift(self, detectors):
        # time shift signals based of detector choice
        ifo1, ifo2 = detectors
        dt = ifo1.light_travel_time_to_detector(ifo2)
        return dt
    
    def add_zero_padding(self, signal, start_time, sample_length, sample_rate):
        # if random duration less than sample_length, add zero padding
        left_pad = int(start_time * sample_rate)
        right_pad = int((sample_length*sample_rate - (left_pad + len(signal))))
        padded_signal = np.pad(signal, (left_pad, right_pad), 'constant', constant_values=(0, 0))

        return padded_signal

    def add_whiten_padding(self, signal, special):
        # whiten padding added separately for ease of understanding
        padding = special['data_cfg'].whiten_padding
        left_pad = right_pad = int((padding/2.) * special['data_cfg'].sample_rate)
        padded_signal = np.pad(signal, (left_pad, right_pad), 'constant', constant_values=(0, 0))
        return padded_signal

    def testing_spectral_bias(self, special):
        ## Generating sin waves with different frequencies but same duration
        # Params
        detectors = special['dets']
        sample_length = special['data_cfg'].signal_length # seconds
        sample_rate = special['data_cfg'].sample_rate # Hz
        # Simulating bias
        random_freq = np.random.uniform(low=self.lower_freq, high=self.upper_freq)
        tseries = np.linspace(0.0, self.fixed_duration, int(self.fixed_duration*sample_rate))
        # Get time shift between detectors
        dt = self.get_time_shift(detectors)
        signal = self.generate(random_freq, tseries)
        start_time = np.random.uniform(self.inject_lower, self.inject_upper)
        signal_det1 = self.add_zero_padding(signal, start_time, sample_length, sample_rate)
        # Add dt to start time for detector offset
        signal_det2 = self.add_zero_padding(signal, start_time, sample_length, sample_rate)
        # Add whiten padding separately
        if not self.no_whitening:
            signal_det1 = self.add_whiten_padding(signal_det1, special)
            signal_det2 = self.add_whiten_padding(signal_det2, special)
        return np.stack((signal_det1, signal_det2), axis=0)

    def testing_duration_bias(self, special):
        ## Generating sin waves with different duration but same frequency
        # Params
        detectors = special['dets']
        sample_length = special['data_cfg'].signal_length # seconds
        sample_rate = special['data_cfg'].sample_rate # Hz
        # Simulating bias
        random_dur = np.random.uniform(low=self.lower_tau, high=self.upper_tau)
        tseries = np.linspace(0.0, random_dur, int(random_dur*sample_rate))
        # Get time shift between detectors
        dt = self.get_time_shift(detectors)
        signal = self.generate(self.fixed_frequency, tseries)
        start_time = np.random.uniform(self.inject_lower, self.inject_upper)
        signal_det1 = self.add_zero_padding(signal, start_time, sample_length, sample_rate)
        signal_det2 = self.add_zero_padding(signal, start_time+dt, sample_length, sample_rate)
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


class FastGenerateWaveform():
    ## Used to augment on all parameters (uses GPU-accelerated IMRPhenomPv2 waveform generation)
    ## Link to Ripple: https://github.com/tedwards2412/ripple
    def __init__(self, 
                 rwrap = 3.0, 
                 beta_taper = 8, 
                 pad_duration_estimate = 1.1, 
                 min_mass = 5.0,
                 one_signal_params = None,
                 debug_me = False
                ):

        # Generate the frequency grid (default values)
        self.f_lower = 0.0 # Hz
        self.f_upper = 0.0 # Hz
        self.delta_t = 0.0 # seconds
        self.f_ref = 0.0 # Hz
        self.sample_rate = 0.0 # Hz
        # Clean-up params
        self.rwrap = rwrap
        # Tapering params
        self.beta = beta_taper
        # Condition for optimising f_min
        self.duration_padfactor = pad_duration_estimate
        # Projection params
        self.signal_length = 0.0 # seconds
        self.whiten_padding = 0.0 # seconds
        self.error_padding_in_s = 0.0 # seconds
        # Other        
        self.min_mass = min_mass
        self.debug_me = debug_me
        # One-Signal mode
        # Whatever the params received from datasets obj
        # use one_signal_params for every iteration
        self.one_signal_params = one_signal_params

    def precompute_common_params(self):
        # Pick the longest waveform from priors to make some params static
        # Default params are for m1 = 5.01, m2 = 5.0 and with aligned spins s1z, s2z = 0.99
        # Minimum mass is chosen to be below prior minimum mass (just in case)
        ## Sanity check
        assert any([self.f_lower, self.f_upper, self.delta_t, \
                    self.f_ref, self.signal_length, self.whiten_padding])
        ## End
        _theta = {'mass1': self.min_mass+0.01, 'mass2': self.min_mass, 'spin1z': 0.99, 'spin2z': 0.99}
        self.tmp_f_lower, self.tmp_delta_f, self.fsize = self.optimise_fmin(_theta)
        # Get the fseries over which we get the waveform in FD
        self.fseries = np.arange(0.0, self.f_upper, self.tmp_delta_f)
        # self.fseries = np.arange(self.tmp_f_lower, self.f_upper, self.tmp_delta_f)
        fseries_trunc = self.fseries[:self.fsize]
        self.cshift = np.exp(-2j*np.pi*(-self.rwrap)*fseries_trunc)
        self.clean_idx = self.fseries < self.tmp_f_lower
        # Windowing params
        self.width = self.f_lower - self.tmp_f_lower
        self.winlen = int(2. * (self.width / self.tmp_delta_f))
        self.window = np.array(get_window(('kaiser', self.beta), self.winlen))
        self.kmin = int(self.tmp_f_lower / self.tmp_delta_f)
        self.kmax = self.kmin + self.winlen//2

    """ ONE-OFF FUNCTIONS """
    def get_imr_duration(self, theta, f_lower):
        # This function is applicable for IMRPhenomD and IMRPhenomPv2
        # Multiplying by a factor of 1.1 for overestimate of signal duration
        return 1.1 * lalsim.SimIMRPhenomDChirpTime(theta['mass1']*1.989e+30, theta['mass2']*1.989e+30, 
                                                   theta['spin1z'], theta['spin2z'], 
                                                   f_lower)
    
    def nearest_larger_binary_number(self, input_len):
        # Return the nearest binary number larger than input_len.
        return int(2**np.ceil(np.log2(input_len)))

    def optimise_fmin(self, theta):
        ## NOTE: We find that even for the longest duration waveform we deal with
        ## the value of f_lower is still 17.02 Hz (we can fix this value and remove this function)
        # determine the duration to use
        full_duration = duration = self.get_imr_duration(theta, self.f_lower)
        tmp_f_lower = self.f_lower
        while True:
            # This iteration is typically done 16 times
            full_duration = self.get_imr_duration(theta, tmp_f_lower)
            condition = duration * self.duration_padfactor
            if full_duration >= condition:
                break
            else:
                # We can change this to tmp_f_lower -= 3.0 to lower iterations
                # It will consequently increase the time taken for waveform generation process
                # But, we've already seen that this shouldn't matter much for Ripple
                # tmp_f_lower *= 0.99 is consistent with PyCBC docs
                tmp_f_lower *= 0.99

        # factor to ensure the vectors are all large enough. We don't need to
        # completely trust our duration estimator in this case, at a small
        # increase in computational cost
        fudge_duration = (full_duration + .1 + self.rwrap) * self.duration_padfactor
        fsamples = int(fudge_duration / self.delta_t)
        N = self.nearest_larger_binary_number(fsamples)
        fudge_duration = N * self.delta_t

        tmp_delta_f = 1.0 / fudge_duration
        tsize = int(1.0 / self.delta_t /  tmp_delta_f)
        fsize = tsize // 2 + 1

        return (tmp_f_lower, tmp_delta_f, fsize)
    
    """ JITTABLES """
    # Jitting these functions require the first argument (self) to be defined as static
    def convert_to_timeseries(self, hpol):
        ## Convert frequency series to time series
        return np.fft.irfft(hpol) * (1./self.delta_t)
    
    def fd_taper_left(self, out):
        # Apply Tapering
        out[self.kmin:self.kmax] = out[self.kmin:self.kmax] * self.window[:self.winlen//2]
        out[:self.kmin] = out[:self.kmin] * 0.
        # Convert frequency series to time series
        out = self.convert_to_timeseries(out)
        return out
    
    def cyclic_time_shift(self, hpol):
        return hpol * self.cshift

    def resize(self, hpol):
        return hpol[0:self.fsize]
    
    def get_theta_pycbc(self, theta):
        # Add required params to waveform kwargs
        theta['f_lower'] = self.tmp_f_lower
        theta['delta_f'] = self.tmp_delta_f
        theta['delta_t'] = self.delta_t
        theta['f_final'] = self.f_upper
        return theta
    
    def get_pycbc_hphc(self, theta):
        # Get frequency domain waveform
        return pycbc.waveform.get_fd_waveform(**theta)

    """ MAIN """
    def make_injection(self, hp, hc, params):
        # Get the required sample length and tc
        sample_length_in_s = len(hp)/self.sample_rate
        tc_obs = sample_length_in_s - self.rwrap
        tc_req = params['tc']
        start_time = tc_obs - tc_req
        end_time = tc_obs + (self.signal_length - tc_req)
        # Pad the start and end times for whitening and error padding
        start_time -= (self.whiten_padding/2.0 + self.error_padding_in_s)
        end_time += (self.whiten_padding/2.0 + self.error_padding_in_s)
        # Pad hpol with zeros (append or prepend) if necessary
        left_pad = int(-start_time * self.sample_rate) if start_time < 0.0 else 0
        right_pad = int((end_time-sample_length_in_s) * self.sample_rate) if end_time > sample_length_in_s else 0
        hp = np.pad(hp, (left_pad, right_pad), 'constant', constant_values=(0.0, 0.0))
        hc = np.pad(hc, (left_pad, right_pad), 'constant', constant_values=(0.0, 0.0))
        # Slice the required section out of hpol
        start_idx = int(start_time*self.sample_rate) if start_time > 0.0 else 0
        end_idx = int(end_time*self.sample_rate) + int(left_pad)
        slice_idx = slice(start_idx, end_idx)
        hp = hp[slice_idx]
        hc = hc[slice_idx]
        return (hp, hc)

    def project(self, hp, hc, special, params):
        # Get hp, hc in the time domain and convert to h(t)
        # Time of coalescence
        tc = params['tc']
        tc_gps = params['injection_time']
        ## Get random value (with a given prior) for polarisation angle, ra, dec
        # np.random.seed(special['sample_seed'])
        # Polarisation angle
        pol_angle = special['distrs']['pol'].rvs()[0][0]
        # Right ascension, declination
        sky_pos = special['distrs']['sky'].rvs()[0]
        declination, right_ascension = sky_pos
        # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
        hp = TimeSeries(hp, delta_t=self.delta_t)
        hc = TimeSeries(hc, delta_t=self.delta_t)
        # Get start interval and end interval for time series
        # Start and end interval define slice of ts without error padding
        pre_coalescence = tc + (self.whiten_padding/2.0)
        start_interval = tc_gps - pre_coalescence
        post_merger = self.signal_length - tc
        end_interval = tc_gps + post_merger + (self.whiten_padding/2.0)
        # Setting the start time for hp and hc
        hp.start_time = hc.start_time = start_interval - self.error_padding_in_s
        # Project wave and find strains for detectors
        strains = [det.project_wave(hp, hc, right_ascension, declination, pol_angle) for det in self.dets]
        time_interval = (start_interval, end_interval)
        strains = np.array([strain.time_slice(*time_interval, mode='nearest') for strain in strains])
        return strains

    def generate(self, _theta):
        theta = _theta.copy()
        ## Generate waveform on the fly using GPU-accelerated Ripple
        # Convert theta to theta_ripple (jnp) (only required params)
        theta_pycbc = self.get_theta_pycbc(theta)
        # Get h_plus and h_cross from the given waveform parameters theta
        # Note than hp and hc are in the frequency domain
        hp, hc = self.get_pycbc_hphc(theta_pycbc)
        # Resizing (to the required sample rate)
        hp = self.resize(hp)
        hc = self.resize(hc)
        # Cyclic time-shift
        hp = self.cyclic_time_shift(hp)
        hc = self.cyclic_time_shift(hc)
        # Tapering and convert from freq domain to time domain (fd_to_td)
        hp_td = self.fd_taper_left(hp)
        hc_td = self.fd_taper_left(hc)

        return hp_td, hc_td
    
    def debug_waveform_generate(self, data, labels, special):
        # Plotting debug recoloured
        cfg = special['cfg']
        # NOTE to self: figsize is (width, height)
        fig, axs = plt.subplots(len(labels), 1, figsize=(9.0, 9.0*len(labels)), squeeze=False)
        fig.suptitle('Debugging Waveform Generation Module')
        for n, (d, l) in enumerate(zip(data, labels)):
            # Subplot top
            axs[n][0].plot(d, label=l)
            axs[n][0].grid()
            axs[n][0].legend()
        # Other
        filename = 'waveform_{}.png'.format(uuid.uuid4().hex)
        dirname = 'training' if special['training'] else 'validation'
        epoch = special['epoch']
        save_path = os.path.join(cfg.export_dir, "DEBUG/waveform_generation/{}/{}".format(epoch, dirname))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save = os.path.join(save_path, filename)
        plt.savefig(save)
        plt.close()
    
    def apply(self, params: dict, special: dict):
        # Set lal.Detector object as global as workaround for MP methods
        # Project wave does not work with DataLoader otherwise
        setattr(self, 'dets', special['dets'])
        ## Augmentation on all params
        if self.one_signal_params != None:
            params = self.one_signal_params
        hp, hc = self.generate(params)
        ## Make hp, hc into proper injection (adjust to tc and zero pad)
        hp, hc = self.make_injection(hp, hc, params)
        ## Convert hp, hc into h(t) using antenna pattern (H1, L1 considered)
        out = self.project(hp, hc, special, params)
        ## Debug waveform generation
        if self.debug_me:
            self.debug_waveform_generate(data=[out[0], out[1]],
                                         labels=['H1', 'L1'],
                                         special=special)
        ## Input: (h_plus, h_cross) --> output: (det1 h_t, det_2 h_t)
        return out
    


""" Signal only Transformations """


class AugmentPolSky(SignalWrapper):
    """ Used to augment polarisation angle, ra and dec (Sky position) """
    def __init__(self, always_apply=True, augmentation=True):
        super().__init__(always_apply)
        self.augmentation = augmentation

    def augment(self, signals, params, special):
        ## Get random value (with a given prior) for polarisation angle, ra, dec
        # Polarisation angle
        pol_angle = special['distrs']['pol'].rvs()[0][0] if self.augmentation else params['polarization']
        # Right ascension, declination
        sky_pos = special['distrs']['sky'].rvs()[0] if self.augmentation else (params['dec'], params['ra'])
        # Times
        time_interval = (params['interval_lower'], params['interval_upper'])
        start_time = params['start_time']
        # h+ and hx
        h_plus = signals[0]
        h_cross = signals[1]

        declination, right_ascension = sky_pos

        # Using PyCBC project_wave to get h_t from h_plus and h_cross
        # Setting the start_time is important! (too late, too early errors are because of this)
        h_plus = TimeSeries(h_plus, delta_t=1./params['sample_rate'])
        h_cross = TimeSeries(h_cross, delta_t=1./params['sample_rate'])
        # Set start times
        h_plus.start_time = start_time
        h_cross.start_time = start_time
        
        # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
        strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle) for det in self.dets]
        # Put both strains together
        augmented_signal = np.array([strain.time_slice(*time_interval, mode='nearest') for strain in strains])
        # Update params
        params['declination'] = declination
        params['right_ascension'] = right_ascension
        params['polarisation_angle'] = pol_angle
        
        return (augmented_signal, params)

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Set lal.Detector object as global as workaround for MP methods
        # Project wave does not work with DataLoader otherwise
        setattr(self, 'dets', special['dets'])
        # Augmentation on polarisation and sky position
        out, params = self.augment(y, params, special)
        # Update params
        params.update(params)
        # Input: (h_plus, h_cross) --> output: (det1 h_t, det_2 h_t)
        # Shape remains the same, so reading in dataset object won't be a problem
        return (out, params, special)


class AugmentDistance(SignalWrapper):
    """ Used to augment the distance parameter of the given signal """
    def __init__(self, always_apply=True, uniform_dchirp=False):
        super().__init__(always_apply)
        self.uniform_dchirp = uniform_dchirp

    def get_augmented_signal(self, signal, params, distrs, debug):
        # Get old params
        distance_old = params['distance']
        mchirp = params['mchirp']
        # Getting new distance
        if self.uniform_dchirp:
            chirp_distance = np.random.uniform(130., 350., size=1)[0]
        else:
            chirp_distance = distrs['dchirp'].rvs()[0][0]
        # Producing the new distance with the required priors
        distance_new = chirp_distance * (2.**(-1./5) * 1.4 / mchirp)**(-5./6)
        
        ## Augmenting on the distance
        augmented_signal = (distance_old/distance_new) * signal
        # Update params
        params['distance'] = distance_new
        params['dchirp'] = chirp_distance
        return (augmented_signal, params)

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Augmenting on distance parameter
        # Unpack required elements from special for augmentation
        distrs = special['distrs']
        norms = special['norm']
        # Run through the augmentation procedure with given dist, mchirp
        out, params = self.get_augmented_signal(y, params, distrs, debug)
        # Update params
        params.update(params)
        # Update special
        special['norm_dist'] = norms['dist'].norm(params['distance'])
        special['norm_dchirp'] = norms['dchirp'].norm(params['dchirp'])
        # Send back the rescaled signal and updated dicts
        return (out, params, special)


class AugmentOptimalNetworkSNR(SignalWrapper):
    """ Used to augment the SNR distribution of the dataset """
    def __init__(self, always_apply=True, rescale=True, 
                 use_uniform=False, 
                 use_beta=False, a=2, b=5,
                 use_add5=False,
                 use_halfnorm=False,
                 snr_lower_limit=5.0,
                 snr_upper_limit=15.0,
                 fix_snr=None):
        
        super().__init__(always_apply)
        # If rescale is False, AUG method returns original network_snr, norm_snr and signal
        self.rescale = rescale
        # Applying a custom distributions for SNR PDFs
        self.use_uniform = use_uniform
        self.use_beta = use_beta
        self.use_add5 = use_add5
        self.a = a
        self.b = b
        self.use_halfnorm = use_halfnorm
        self.snr_lower_limit = snr_lower_limit
        self.snr_upper_limit = snr_upper_limit
        self.fix_snr = fix_snr

    def _dchirp_from_dist(self, dist, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the chirp distance given the luminosity distance and chirp mass.
        return dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)

    def get_rescaled_signal(self, signal, psds, params, cfg, debug, training, aux, epoch, seed):
        # params: This will contain params var found in __getitem__ method of custom dataset object
        # np.random.seed(seed) ----------------------------------------------------------------------------------- ???
        # Get original network SNR
        prelim_network_snr = get_network_snr(signal, psds, params, cfg.export_dir, debug)
        
        if self.rescale or not training:
            if aux == -1:
                # Rescaling the SNR to a uniform distribution within a given range
                rescaled_snr_lower = self.snr_lower_limit
                rescaled_snr_upper = self.snr_upper_limit
                    
                # Uniform on SNR range
                if self.use_uniform:
                    target_snr = np.random.uniform(rescaled_snr_lower, rescaled_snr_upper)
                elif self.use_beta:
                    target_snr = beta.rvs(self.a, self.b)
                    target_snr *= rescaled_snr_upper
                    target_snr += rescaled_snr_lower
                elif self.use_add5:
                    # Make everything detectible
                    target_snr = prelim_network_snr + 5.0
                elif self.use_halfnorm:
                    target_snr = halfnorm.rvs() * 4.0 + 5.0

            elif aux in [0, 2]:
                target_snr = 5.0
            elif aux in [1, 3]:
                target_snr = 12.0
            else:
                raise ValueError('Unidentified value for cflag!')
            
            # Fix SNR for all input signals
            if self.fix_snr != None:
                target_snr = self.fix_snr
            
            rescaling_factor = target_snr/prelim_network_snr
            # Add noise to rescaled signal
            rescaled_signal = signal * rescaling_factor
            
            # Adjust distance parameter for signal according to the new rescaled SNR
            rescaled_distance = params['distance'] / rescaling_factor
            rescaled_dchirp = self._dchirp_from_dist(rescaled_distance, params['mchirp'])
            # Update targets and params with new rescaled distance is not possible
            # We do not know the priors of network_snr properly
            if 'parameter_estimation' in cfg.model_params.keys():
                parameter_estimation = cfg.model_params['parameter_estimation']
                if 'norm_dist' in parameter_estimation or 'norm_dchirp' in parameter_estimation:
                    raise RuntimeError('rescale_snr option cannot be used with dist/dchirp PE!')
            # Update the params dictionary with new rescaled distances
            params['distance'] = rescaled_distance
            params['dchirp'] = rescaled_dchirp
            # Add network snr to params as well
            params['network_snr'] = target_snr
        else:
            # Default option returns only network snr
            params['network_snr'] = prelim_network_snr
            rescaled_signal = signal

        return (rescaled_signal, params)

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Unpack required elements from special for augmentation
        psds = special['psds']
        cfg = special['cfg']
        training = special['training']
        aux = special['aux']
        norms = special['norm']
        epoch = special['epoch']
        seed = special['sample_seed']
        # Augmentation on optimal network SNR
        out, params = self.get_rescaled_signal(y, psds, params, cfg, debug, training, aux, epoch, seed)
        # Update params
        params.update(params)
        # Update special
        special['network_snr'] = params['network_snr']
        special['norm_snr'] = norms['snr'].norm(params['network_snr'])
        # Send back the rescaled signal and updated dicts
        return (out, params, special)



""" Noise only Transformations """

class CyclicShift(NoiseWrapper):
    """ Used to cyclic shift the noise (can be applied to real noise as well) """
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, debug=''):
        # Cyclic shifting noise is possible for artificial and real noise
        num_roll = [random.randint(0, y.shape[1]), random.randint(0, y.shape[1])]
        augmented_noise = np.array([np.roll(foo, num_roll[n]) for n, foo in enumerate(y)])
        return augmented_noise


class AugmentPhase(NoiseWrapper):
    """ Used to augment the phase of noise (can be applied to real noise as well) """
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, debug=''):
        # Phase augmentation of noise is possible for artificial and real noise
        dt = [random.randint(0, y.shape[1])/2048., random.randint(0, y.shape[1])/2048.]
        dphi = np.random.uniform(0.0, 2.0*np.pi, 1)[0]
        # Apply the phase augmentation and time shift in frequency domain
        noise_fft = [np.fft.rfft(foo) for foo in y]
        # TODO: Assuming sampling frequency here. Generalise this.
        df = 1./(y.shape[1]/2048.)
        f = np.arange(len(noise_fft[0])) * df
        # We have turned off shift in time until properly debugged (March 16th, 2023)
        # augmented_noise_fft = [foo*np.exp(2j*np.pi*f*dt[n] + 1j*dphi) for n, foo in enumerate(noise_fft)]
        # NOTE: We are excluding time shifts here for now. They seem to introduce artifacts in the noise after whitening.
        augmented_noise_fft = [foo * np.exp(1j*dphi) for n, foo in enumerate(noise_fft)]
        # Go back to time domain and we should have augmented noise
        augmented_noise = np.array([np.fft.irfft(foo) for foo in augmented_noise_fft])

        return augmented_noise


class Recolour(NoiseWrapper):
    """ Used to augment the PSD of given real noise segment (D4) """
    # This method required extra sample length for noise (equal to whiten_padding). 
    # Whitening module removes corrupted bits due to possible edge effects.
    def __init__(self, always_apply=True,
                 use_precomputed=False, h1_psds_hdf="", l1_psds_hdf="",
                 use_shifted=False, shift_up_factor=10, shift_down_factor=1,
                 p_recolour=0.3,
                 trunc_method='hann',
                 debug_me=False,
                 debug_dir=""):
        
        super().__init__(always_apply)
        # Warnings
        # Using precomputed PSDs. Make sure that they do not include PSDs from testing dataset.
        # Unless using a robust set of PSDs, this method is bound by the limitations of supervised learning
        
        # Using precomputed (using 81 days of O3a noise)
        # WARNING: This is a cheaty method if this uses testing data PSDs for training.
        self.use_precomputed = use_precomputed
        self.h1_psds = h5py.File(h1_psds_hdf, 'r')
        self.shape_h1_psds = dict(self.h1_psds.attrs)['shape']
        self.l1_psds = h5py.File(l1_psds_hdf, 'r')
        self.shape_l1_psds = dict(self.h1_psds.attrs)['shape']
        # Using shifted PSD (shift along y axis)
        self.use_shifted = use_shifted
        self.shift_up_factor = shift_up_factor
        self.shift_down_factor = shift_down_factor
        # Probability of being recoloured
        self.p_recolour = p_recolour

        # Other params
        self.fs = 2048. #Hz
        self.sample_length_in_s = 0.0 # seconds
        self.noise_low_freq_cutoff = 0.0 # Hz
        self.signal_low_freq_cutoff = 0.0 # Hz
        self.whiten_padding = 0.0 # seconds
        self.trunc_method = trunc_method

        # DEBUGGER
        self.debug_me = debug_me
        if debug_me:
            # TODO: If this is fast enough, include in export dir
            self.debug_dir = debug_dir
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
            save_txt = os.path.join(debug_dir, 'recolour.txt')
            self.tmp_debug = open(save_txt, "a")
    
    def estimate_psd(self, ts, DET):
        # Compute PSD using Welch's method
        if DET['is_recolour']:
            freqs, psd = scipy_welch(ts, fs=self.fs, nperseg=4.*self.fs, average='median')
            delta_f = freqs[1]-freqs[0]
            DET['old_psd'] = psd
            DET['old_delta_f'] = delta_f
        return DET
    
    def inv_spec_trunc(self, psd, max_filter_len):
        # Interpolate and smooth to the desired corruption length
        psd = inverse_spectrum_truncation(psd,
                                        max_filter_len=max_filter_len,
                                        low_frequency_cutoff=self.signal_low_freq_cutoff,
                                        trunc_method=self.trunc_method)
        return psd

    def whiten(self, signal, psd, signal_delta_f):
        """ Return a whitened time series """
        # Convert signal to Timeseries object
        signal = TimeSeries(signal, delta_t=1./self.fs)
        # Filter length for inverse spectrum truncation
        max_filter_len = int(round(self.whiten_padding * self.fs))
        ## Manipulate PSD for usage in whitening
        delta_f = signal_delta_f
        ## Whitening
        psd = self.inv_spec_trunc(psd, max_filter_len)
        # NOTE: Factor of dt not taken into account. Since normlayer takes care of standardisation,
        # we don't necessarily need to include this. After decorrelation, the diagonal matrix 
        # values will not be 1's but some other value dependant on the signal input.
        white_frequency_series = (signal.to_frequencyseries(delta_f=delta_f) / psd**0.5)
        return (white_frequency_series, max_filter_len)

    def get_psd(self, H1, L1):
        ## Use pre-computed PSDs from HDF5 file
        idx_h1 = -1
        idx_l1 = -1
        # H1 - use different PSD
        if H1['is_diff_psd'] and H1['is_recolour']:
            idx_h1 = np.random.randint(0, int(self.shape_h1_psds[0]))
            H1['new_psd'] = self.h1_psds['data'][idx_h1]
        else:
            H1['new_psd'] = None # H1['old_psd']
        # L1 - use different PSD
        if L1['is_diff_psd'] and L1['is_recolour']:
            idx_l1 = np.random.randint(0, int(self.shape_l1_psds[0]))
            L1['new_psd'] = self.l1_psds['data'][idx_l1]
        else:
            L1['new_psd'] = None # L1['old_psd']

        # Debugger
        if self.debug_me:
            foo = "{}, {}, {}, {}".format(idx_h1, idx_l1, H1['is_recolour'], L1['is_recolour'])
            self.tmp_debug.write(foo)

        return H1, L1

    def shift_psd(self, H1, L1):
        # Shift new PSD along y axis
        H1_shift_up_factor = np.random.uniform(1, self.shift_up_factor)
        H1_shift_down_factor = np.random.uniform(1, self.shift_down_factor)**-1
        H1_up_or_down = 1 if np.random.random() < 0.5 else 0
        if H1['is_recolour']:
            H1['new_psd'] *= H1_shift_up_factor if H1_up_or_down else H1_shift_down_factor
        L1_shift_up_factor = np.random.uniform(1, self.shift_up_factor)
        L1_shift_down_factor = np.random.uniform(1, self.shift_down_factor)**-1
        L1_up_or_down = 1 if np.random.random() < 0.5 else 0
        if L1['is_recolour']:
            L1['new_psd'] *= L1_shift_up_factor if L1_up_or_down else L1_shift_down_factor
        return (H1, L1)
    
    def debug_recolour(self, data, labels):
        # Plotting debug recoloured
        # NOTE to self: figsize is (width, height)
        fig, axs = plt.subplots(len(labels), 1, figsize=(9.0, 9.0*len(labels)), squeeze=False)
        fig.suptitle('Debugging Recolour Module')
        for n, (d, l) in enumerate(zip(data, labels)):
            # Subplot top
            if 'psd' in l:
                axs[n][0].loglog(d, label=l)
            else:
                axs[n][0].plot(d, label=l)
            axs[n][0].grid()
            axs[n][0].legend()
        # Other
        filename = 'recolour_{}.png'.format(uuid.uuid4().hex)
        save = os.path.join(self.debug_dir, filename)
        plt.savefig(save)
        plt.close()
    
    def resize_to_samplelen(self, ts):
        crop = slice(int(self.whiten_padding/2.*self.fs), -int(self.whiten_padding/2.*self.fs))
        cropped = ts[crop]
        return cropped

    def recolour(self, ts, DET):
        ## Whiten the noise using old PSD and recolour using new PSD
        if not DET['is_recolour']:
            cropped = self.resize_to_samplelen(ts)
            return cropped
        # Add a whiten padding to either side of the ts (will be corrupted)
        # padlen = int((self.whiten_padding/2.0)*self.fs)
        # ts = np.pad(ts, (padlen, padlen), 'constant', constant_values=(0, 0))
        # delta_f will have to be changed based on new length
        data_delta_f = 1./(self.sample_length_in_s+self.whiten_padding)
        # Convert the PSD to new delta_f using PyCBC interpolate function
        old_psd = FrequencySeries(DET['old_psd'], delta_f=DET['old_delta_f'])
        old_psd = interpolate(old_psd, data_delta_f)
        # Whitening (Remove old PSD from data) still in fd
        whitened_fd, max_filter_len = self.whiten(ts, old_psd, data_delta_f)
        # Convert the new PSDs to have delta_f similar to data
        # new_psd = FrequencySeries(DET['new_psd'], delta_f=0.25)
        new_psd = FrequencySeries(DET['new_psd'], delta_f=DET['old_delta_f'])
        new_psd = interpolate(new_psd, data_delta_f)
        new_psd = self.inv_spec_trunc(new_psd, max_filter_len)
        # Recolour using new PSD and return to time domain
        recoloured = (whitened_fd * new_psd**0.5).to_timeseries()
        # NOTE: Removing 5 seconds of data here. Make sure sample length is set accordingly.
        recoloured = recoloured[int(max_filter_len/2):int(len(recoloured)-max_filter_len/2)].numpy()
        # debug plotter
        if self.debug_me:
            _, recovered = scipy_welch(recoloured, fs=self.fs, nperseg=4.*self.fs, average='median')
            self.debug_recolour([old_psd, new_psd, ts, recoloured, recovered],
                                ['old_psd', 'new_psd', 'original', 'recoloured', 'recovered_psd'])

        return recoloured

    def apply(self, y: np.ndarray, debug=''):
        # Apply given PSD augmentation technique
        ## Or not to recolour
        if np.random.rand() >= self.p_recolour:
            cropped_h1 = self.resize_to_samplelen(y[0])
            cropped_l1 = self.resize_to_samplelen(y[1])
            y = np.stack([cropped_h1, cropped_l1], axis=0)
            return y
        ## To Recolour
        # Is the detector noise going to be recoloured?
        # Is the detector PSD going to be shifted along y axis?
        # Is the detector PSD going to be blurred with Gaussian noise?
        always_recolour = True if self.p_recolour == 1.0 else False
        H1 = {'is_recolour': np.random.rand() < 0.5,
              'is_diff_psd': np.random.rand() < 0.5,
              'is_shifted': np.random.rand() < 0.5,
              'is_blurred': np.random.rand() < 0.5,
              'recoloured': y[0]}
        L1 = {'is_recolour': np.random.rand() < 0.5,
              'is_diff_psd': np.random.rand() < 0.5,
              'is_shifted': np.random.rand() < 0.5,
              'is_blurred': np.random.rand() < 0.5,
              'recoloured': y[1]}
        # Check for always recolour
        if always_recolour:
            H1['is_recolour'] = True
            L1['is_recolour'] = True
        # Sanity check (happens 25% of the time)
        if not H1['is_recolour'] and not L1['is_recolour']:
            cropped_h1 = self.resize_to_samplelen(y[0])
            cropped_l1 = self.resize_to_samplelen(y[1])
            y = np.stack([cropped_h1, cropped_l1], axis=0)
            return y
        # shifted and blurred are not implemented yet
        # TODO: Remove this after implementation
        H1['is_diff_psd'] = H1['is_recolour']
        L1['is_diff_psd'] = L1['is_recolour']
        # Estimate the old PSD of each detector (as required)
        H1 = self.estimate_psd(y[0], H1)
        L1 = self.estimate_psd(y[1], L1)
        # Recolour and augment (as required)
        if self.use_precomputed:
            H1, L1 = self.get_psd(H1, L1)
        if self.use_shifted:
            H1, L1 = self.shift_psd(H1, L1)
        
        # Adjusting H1 and L1 for extra padding added (if needed)
        H1['recoloured'] = self.recolour(y[0], H1)
        L1['recoloured'] = self.recolour(y[1], L1)

        recoloured_noise = np.stack([H1['recoloured'], L1['recoloured']], axis=0)
        return recoloured_noise



""" Noise Generation """

class GlitchAugmentGWSPY():
    """ 
    Used to augment the noise samples using GWSPY glitch GPS times 
        1. We download 60 seconds of data from GWOSC per glitch
        2. 30 seconds on either side of event time
        3. Make sure to NOT to use O3a glitches when testing on O3a data
        4. Pick a random start time (appropriate based on sample len) within the segment
        5. We have about 280,000 segments in O3b = 19.44 days
        6. Using a random start time (aka time slides) gives us more realisations
    
    """
    def __init__(self,
                 glitch_dirs=dict(
                    H1_O3a="/local/scratch/igr/nnarenraju/gwspy/H1_O3a_glitches",
                    L1_O3a="/local/scratch/igr/nnarenraju/gwspy/L1_O3a_glitches",
                    H1_O3b="/local/scratch/igr/nnarenraju/gwspy/H1_O3b_glitches",
                    L1_O3b="/local/scratch/igr/nnarenraju/gwspy/L1_O3b_glitches"
                 ),
                 include=['H1_O3a', 'H1_O3b', 'L1_O3a', 'L1_O3b'],
                 debug_me=False,
                 debug_dir=""):
        
        # Assertions
        assert len(include) > 0, 'At least one observing run for a detector must be selected'

        # Glitch data files
        self.include = include
        self.glitch_files = {}
        self.num_glitches = {}
        for name in include:
            self.glitch_files[name] = [h5py.File(fname) for fname in glob.glob(os.path.join(glitch_dirs[name], "*.hdf"))]
            self.num_glitches[name] = np.load("./notebooks/tmp/{}.npy".format(name))
        
        # DEBUGGER
        self.debug_me = debug_me
        if debug_me:
            self.debug_dir = debug_dir
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
            save_txt = os.path.join(debug_dir, 'gravity_spy.txt')
            self.tmp_debug = open(save_txt, "a")

    def pick_glitch_file_H1(self, obs_run):
        # Pick a glitch file for each detector
        # Glitch sample obtained from GWOSC using GWSPY GPS times
        H1opt = ([self.glitch_files['H1_O3a'], self.num_glitches['H1_O3a']] 
                  if obs_run=='H1_O3a' 
                  else [self.glitch_files['H1_O3b'], self.num_glitches['H1_O3b']])
        # Pick a random file to get the glitch
        idx = np.random.choice(list(range(len(H1opt[1]))))
        H1_file, H1_file_len = H1opt[0][idx], H1opt[1][idx]
        return H1_file, H1_file_len
    
    def pick_glitch_file_L1(self, obs_run):
        # Pick a glitch file for each detector
        # Glitch sample obtained from GWOSC using GWSPY GPS times
        L1opt = ([self.glitch_files['L1_O3a'], self.num_glitches['L1_O3a']] 
                  if obs_run=='L1_O3a' 
                  else [self.glitch_files['L1_O3b'], self.num_glitches['L1_O3b']])
        # Pick a random file to get glitch
        idx = np.random.choice(list(range(len(L1opt[1]))))
        L1_file, L1_file_len = L1opt[0][idx], L1opt[1][idx]
        return L1_file, L1_file_len

    def read_glitch(self, hf, length, data_cfg, is_training):
        # Check if training (worry about using recolour)
        if is_training:
            recolour_pad = int(data_cfg.whiten_padding*data_cfg.sample_rate)
        else:
            recolour_pad = 0
        # Pick a random glitch from glitch file
        idx = np.random.randint(low=0, high=int(length))
        # Pick a random start time from 0 to 60-sample_length
        start_time_ulimit = int(60.*2048.) - (int(data_cfg.sample_length_in_num)+recolour_pad)
        rand_start_idx = np.random.randint(low=0, high=start_time_ulimit)
        rand_slice = slice(rand_start_idx, int(rand_start_idx+data_cfg.sample_length_in_num+recolour_pad))
        # Get the required segment from glitch array
        glitch = np.array(hf['data'][idx][rand_slice]).astype(np.float64)
        return glitch

    def debug_gwspy(self, data, labels):
        # Debugging plotter for gwspy
        fig, axs = plt.subplots(len(labels), 1, figsize=(9.0, 9.0*len(labels)), squeeze=False)
        fig.suptitle('Debugging Gravity Spy Augmentation')
        for n, (d, l) in enumerate(zip(data, labels)):
            # Subplot
            axs[n][0].plot(d, label=l)
            axs[n][0].grid()
            axs[n][0].legend()
        # Other
        filename = 'gravity_spy_{}.png'.format(uuid.uuid4().hex)
        save = os.path.join(self.debug_dir, filename)
        plt.savefig(save)
        plt.close()

    def apply(self, special):
        ## Get random glitch for detector(s)
        if special['training']:
            recolour_pad = int(special['data_cfg'].whiten_padding*
                               special['data_cfg'].sample_rate)
        else:
            recolour_pad = 0
        # Is the detector going to get a glitch?
        is_glitch = {'H1': np.random.rand() < 0.5, 
                     'L1': np.random.rand() < 0.5}

        # Pick on observing run for each detector (if needed)
        h1_options = [opt for opt in self.include if 'H1' in opt]
        l1_options = [opt for opt in self.include if 'L1' in opt]

        pick_observing_run = {'H1': np.random.choice(h1_options),
                              'L1': np.random.choice(l1_options)}

        # Read the glitch from the provided filenum
        glitch_H1 = np.zeros(int(special['data_cfg'].sample_length_in_num + recolour_pad))
        while True and is_glitch['H1']:
            H1_file, H1_file_len = self.pick_glitch_file_H1(pick_observing_run['H1'])
            glitch_H1 = self.read_glitch(H1_file, H1_file_len, special['data_cfg'], special['training'])
            if not any(np.isnan(glitch_H1)):
                break
        
        glitch_L1 = np.zeros(int(special['data_cfg'].sample_length_in_num + recolour_pad))
        while True and is_glitch['L1']:
            L1_file, L1_file_len = self.pick_glitch_file_L1(pick_observing_run['L1'])
            glitch_L1 = self.read_glitch(L1_file, L1_file_len, special['data_cfg'], special['training'])
            if not any(np.isnan(glitch_L1)):
                break
        
        # Debugging
        if self.debug_me:
            foo = "{}, {}, {}, {}".format(is_glitch['H1'], is_glitch['L1'], 
                                          pick_observing_run['H1'], pick_observing_run['L1'])
            self.tmp_debug.write(foo)
            data = [glitch_H1, glitch_L1]
            labels = ['glitch H1', 'glitch L1']
            self.debug_gwspy(data, labels)

        # Augmented noise (Downsampled to 2048. Hz after downloading)
        noise = np.stack([glitch_H1, glitch_L1], axis=0)
        return noise


class MultipleFileRandomNoiseSlice():
    """ 
    Same as RandomNoiseSlice but for multiple noise files with different durations in each
        1. Downloaded ~113 days of noise from O3b for H1 and L1
        2. PSDs shouldn't vary too drastically from O3a
        3. Each segment is at least 1 hour in length and stored in separate files

    """
    def __init__(self,
                 noise_dirs=dict(
                    H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                    L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                 ),
                 debug_me=False,
                 debug_dir=""):

        # Noise data files
        self.sample_length = 0.0 # seconds
        self.noise_files = {}
        self.lengths = {}
        for name in noise_dirs.keys():
            self.noise_files[name] = [h5py.File(fname) for fname in glob.glob(os.path.join(noise_dirs[name], "*.hdf"))]
            self.lengths[name] = np.load("./notebooks/tmp/durs_{}_O3b_all_noise.npy".format(name)) # _deimos

    def pick_noise_file(self, det):
        # Pick a noise file for each detector
        # Pick a random file to get the noise sample
        idx = np.random.choice(list(range(len(self.lengths[det]))))
        det_file, det_file_length = self.noise_files[det][idx], self.lengths[det][idx]
        return det_file, det_file_length
    
    def _make_sample_start_time(self, seg_start_idx, seg_end_idx):
        # Make a sample start time that is uniformly distributed within segdur
        return int(np.random.uniform(low=seg_start_idx, high=seg_end_idx))

    def read_noise(self, hf, length, data_cfg, recolour_pad):
        # Get random noise segment
        seg_start_idx, seg_end_idx = (0, length-1)
        seg_end_idx -= (recolour_pad + self.sample_length*data_cfg.sample_rate)
        # This start time will lie within a valid segment time interval
        sample_start_idx = self._make_sample_start_time(seg_start_idx, seg_end_idx)
        # Get the required portion of given segment
        sidx = sample_start_idx
        eidx = sample_start_idx + int(self.sample_length * data_cfg.sample_rate)
        eidx += recolour_pad
        # Get time series from segment and apply the dynamic range factor
        ts = np.array(hf['data'][sidx:eidx]).astype(np.float64)
        ts /= DYN_RANGE_FAC
        return ts

    def apply(self, special, det_only=''):
        ## Get random noise sample for detector(s)
        if special['cfg'].transforms['noise'] != None:
            get_class = lambda clist, cname: [foo for foo in clist if foo.__class__.__name__==cname][0]
            recolour = get_class(special['cfg'].transforms['noise'].transforms, 'Recolour')
            recolour_flag = True if recolour != [] else False
        else:
            recolour_flag = False
        
        if special['training']:
            if recolour_flag:
                recolour_pad = int(special['data_cfg'].whiten_padding*special['data_cfg'].sample_rate)
            else:
                recolour_pad = 0
        else:
            recolour_pad = 0
        # Is the detector going to be augmented with extra noise?
        is_augment = {'H1': np.random.rand() < 0.5, 
                      'L1': np.random.rand() < 0.5}

        # Read the noise from the provided filenum
        noise_H1 = np.zeros(int(special['data_cfg'].sample_length_in_num + recolour_pad))
        while True and is_augment['H1']:
            H1_file, H1_file_len = self.pick_noise_file('H1')
            noise_H1 = self.read_noise(H1_file, H1_file_len, special['data_cfg'], recolour_pad)
            if not any(np.isnan(noise_H1)):
                break
        
        noise_L1 = np.zeros(int(special['data_cfg'].sample_length_in_num + recolour_pad))
        while True and is_augment['L1']:
            L1_file, L1_file_len = self.pick_noise_file('L1')
            noise_L1 = self.read_noise(L1_file, L1_file_len, special['data_cfg'], recolour_pad)
            if not any(np.isnan(noise_L1)):
                break

        # Augmented noise (Downsampled to 2048. Hz after downloading)
        noise = np.stack([noise_H1, noise_L1], axis=0)
        return noise
    

class RandomNoiseSlice():
    """ Used to augment the start time of noise samples from continuous noise .hdf file """
    # This will become the primary noise reading function
    def __init__(self, 
                 real_noise_path = "", 
                 segment_llimit = None, 
                 segment_ulimit = None, 
                 debug_me = False
                ):
        
        self.sample_length = 0.0 # seconds
        self.dt = 0.0 # seconds
        self.real_noise_path = real_noise_path
        # Values set using parameters from MLGWSC-1
        self.segment_ends_buffer = 0.0 # seconds
        self.slide_buffer = 240.0
        # Other
        self.segment_llimit = segment_llimit
        self.segment_ulimit = segment_ulimit
        self.debug_me = debug_me
    
    def precompute_common_params(self):
        # Set minimum segment duration
        self.min_segment_duration = self.sample_length # seconds
        # Keep all required noise files open
        self.O3a_real_noise = h5py.File(self.real_noise_path, 'r')
        # Get detectors used
        self.detectors = ['H1', 'L1']
        # Get ligo segments and load_times from noise file
        ligo_segments, load_times = self._get_ligo_segments()
        # Get segment info and set probability of obtaining sample from segment
        self.psegment = {}
        segdurs = np.empty(len(ligo_segments), dtype=np.float64)
        # limit check
        if self.segment_ulimit == -1:
            self.segment_ulimit = len(ligo_segments)
        
        lookup = np.load("./notebooks/tmp/segdurs_all.npy")
        for n, seg in enumerate(ligo_segments):
            key_time = str(load_times[seg][0])
            _key = f'{self.detectors[0]}/{key_time}'
            # Sanity check if _key is present in noise file
            if n >= self.segment_llimit and n <= self.segment_ulimit:
                segment_length = lookup[:,1][n]
                seg_start_idx = 0 + self.segment_ends_buffer
                seg_end_idx = segment_length - (self.sample_length + self.segment_ends_buffer)*(1./self.dt)
                segdurs[n] = lookup[:,1][n]
                self.psegment[n] = [key_time, seg_start_idx, seg_end_idx]
            else:
                # An impossible segment duration and cond rand < segprob is never satisfied
                segdurs[n] = 0
                self.psegment[n] = [-1, -1, -1]

        # Get probabilties of using segment using segment durations
        self.seg_idx = np.arange(len(segdurs))
        self.segprob = list(segdurs/np.sum(segdurs))

        # Debugging
        if self.debug_me:
            save_txt = os.path.join('./tmp', 'random_noise_slice.txt')
            self.tmp_debug = open(save_txt, "a")
    
    def get_segment_choice(self, seed):
        # Get one choice from seg_idx based on probalities obtained from seg durations
        np.random.seed(seed) # ------------------------------------------------------------------------------ REMOVE THIS!!!
        return np.random.choice(self.seg_idx, 1, p=self.segprob)[0]

    def _load_segments(self):
        tmp_dir = "./tmp"
        path = os.path.join(tmp_dir, 'segments.csv')
        # Download data if it does not exist
        if not os.path.isfile(path):
            url = 'https://www.atlas.aei.uni-hannover.de/work/marlin.schaefer/MDC/segments.csv'
            response = requests.get(url)
            with open(path, 'wb') as fp:
                fp.write(response.content)

        # Load data from CSV file
        segs = ligo.segments.segmentlist([])
        with open(path, 'r') as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                idx, start, end = row
                segs.append(ligo.segments.segment([int(start), int(end)]))

        return segs
    
    def _get_ligo_segments(self):
        # https://lscsoft.docs.ligo.org/ligo-segments/
        segments = self._load_segments()
        
        # Restrict segments
        ligo_segments = ligo.segments.segmentlist([])
        for seg in segments:
            start, end = seg
            segduration = end - start
            # Check if segment fulfills minimum duration requirements
            if self.min_segment_duration is not None and segduration - self.slide_buffer < self.min_segment_duration:
                continue
            ligo_segments.append(ligo.segments.segment([start, end]))
        
        # Refer link provided above to ligo-segments
        # Sort the elements of the list into ascending order, and merge continuous 
        # segments into single segments. Segmentlist is modified in place. 
        # This operation is O(n log n).
        ligo_segments.coalesce()

        # Get times from each valid segment
        load_times = {}
        for seg in ligo_segments:
            for rawseg in segments:
                if seg in rawseg:
                    load_times[seg] = rawseg
                    break;
            if seg not in load_times:
                raise RuntimeError
            
        return ligo_segments, load_times
    
    def debug_random_noise_slice(self, data, labels, special):
        # Plotting debug recoloured
        # NOTE to self: figsize is (width, height)
        fig, axs = plt.subplots(len(labels), 1, figsize=(9.0, 9.0*len(labels)), squeeze=False)
        fig.suptitle('Debugging Random Noise Slice')
        for n, (d, l) in enumerate(zip(data, labels)):
            # Subplot top
            axs[n][0].plot(d, label=l)
            axs[n][0].grid()
            axs[n][0].legend()
        # Other
        filename = 'random_noise_slice_{}.png'.format(uuid.uuid4().hex)
        dirname = 'training' if special['training'] else 'validation'
        epoch = special['epoch']
        cfg = special['cfg']
        save_path = os.path.join(cfg.export_dir, "DEBUG/noise_generation_d4/{}/{}".format(epoch, dirname))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save = os.path.join(save_path, filename)
        plt.savefig(save)
        plt.close()
    
    def _make_sample_start_time(self, seg_start_idx, seg_end_idx, seed):
        # Make a sample start time that is uniformly distributed within segdur
        np.random.seed(seed) # ---------------------------------------------------------------------------- REMOVE THIS!!!
        return int(np.random.uniform(low=seg_start_idx, high=seg_end_idx))

    def get_noise_segment(self, special, segdeets, det_only, recolour):
        ## Get noise sample from given O3a real noise segment
        # For validation, recolour is always off
        if special['training']:
            if recolour:
                recolour_pad = int(special['data_cfg'].whiten_padding*special['data_cfg'].sample_rate)
            else:
                recolour_pad = 0
        else:
            recolour_pad = 0

        # Get random noise segment
        rs = np.random.RandomState(seed=special['sample_seed'])
        seeds = list(rs.randint(0, 2**32, len(self.detectors)))
        noise = []
        tmp_key_times = []
        tmp_start_idxs = []
        for det, segdeet, seed in zip(self.detectors, segdeets, seeds):
            if det_only != '' and det_only != det:
                continue
            key_time, seg_start_idx, seg_end_idx = segdeet
            seg_end_idx -= recolour_pad
            # Get sample_start_time using segment times
            # This start time will lie within a valid segment time interval
            sample_start_idx = self._make_sample_start_time(seg_start_idx, seg_end_idx, seed)
            # Get the required portion of given segment
            sidx = sample_start_idx
            eidx = sample_start_idx + int(self.sample_length / self.dt)
            eidx += recolour_pad
            # Which key does the current segment belong to in real noise file
            # key_time provided is the start time of required segment
            key = f'{det}/{key_time}'
            # Update debugging tools
            if self.debug_me:
                tmp_key_times.append(key_time)
                tmp_start_idxs.append(sidx)
            # Get time series from segment and apply the dynamic range factor
            ts = np.array(self.O3a_real_noise[key][sidx:eidx]).astype(np.float64)
            if "O3a_real_noise.hdf" in self.real_noise_path:
                ts /= DYN_RANGE_FAC
            # Send back one det data if requested
            if det_only != '' and det_only == det:
                return ts
            # Else send back stacked noise
            noise.append(ts)
        
        # Debugging
        if self.debug_me:
            self.debug_random_noise_slice(data=noise, labels=['H1 noise', 'L1 noise'], special=special)
            dirname = 'training' if special['training'] else 'validation'
            epoch = special['epoch']
            debug_args = (epoch, dirname, tmp_key_times[0], tmp_start_idxs[0], tmp_key_times[1], tmp_start_idxs[1])
            foo = '{}, {}, H1, {}, {}, L1, {}, {}'.format(*debug_args)
            self.tmp_debug.write(foo)
        # Convert noise into np.ndarray, suitable for other transformations
        noise = np.stack(noise, axis=0)
        return noise
    
    def pick_segment(self, seed):
        # Pick a random segment to use based on probablities set using their duration
        # Picking two different segments and start times provides an extra layer of augmentation
        rs = np.random.RandomState(seed=seed)
        seeds = list(rs.randint(0, 2**32, len(self.detectors)))
        idx1 = self.get_segment_choice(seeds[0])
        idx2 = self.get_segment_choice(seeds[1])
        # Return the segment details of selected segment
        return (self.psegment[idx1], self.psegment[idx2])

    def apply(self, special, det_only=''):
        ## Get noise sample with random start time from O3a real noise
        # Check whether recolour is done
        if special['cfg'].transforms['noise'] != None:
            get_class = lambda clist, cname: [foo for foo in clist if foo.__class__.__name__==cname][0]
            recolour = get_class(special['cfg'].transforms['noise'].transforms, 'Recolour')
            recolour_flag = True if recolour != [] else False
        else:
            recolour_flag = False
        # Toss a biased die and retrieve the segment to use
        segdeets = self.pick_segment(special['sample_seed'])
        # Get noise sample with random start time (uniform within segment)
        noise = self.get_noise_segment(special, segdeets, det_only, recolour_flag)
        # Return noise data
        return noise


class ColouredNoiseGenerator():
    """ Generate Dataset 3 -like noise for Sage training """
    
    def __init__(self, psds_dir: str = ""):
        self.psds_dir = psds_dir
        # H1 and L1 dirs expected inside psds parent directory
        H1_dir = os.path.join(self.psds_dir, 'H1')
        L1_dir = os.path.join(self.psds_dir, 'L1')
        # Get all .hdf files containing one psd each
        self.psd_options = {'H1': glob.glob(os.path.join(H1_dir, '*.hdf')),
                            'L1': glob.glob(os.path.join(L1_dir, '*.hdf'))}
        # Other params
        self.sample_length = None
        self.delta_f = None
        self.noise_low_freq_cutoff = None
        self.sample_rate = None
    
    def precompute_common_params(self):
        # Compute ASD for chosen PSD
        self.complex_asds = {det:[] for det in self.psd_options.keys()}
        for i, det in enumerate(self.psd_options.keys()):
            # Read all detector PSDs as frequency series with appropriate delta_f
            for psd_det in self.psd_options[det]:
                psd = load_frequencyseries(psd_det)
                psd = interpolate(psd, 1.0/self.sample_length)
                # Convert PSD's to ASD's for colouring the white noise
                foo = self.psd_to_asd(psd, 0.0, self.sample_length,
                                sample_rate=self.sample_rate,
                                low_frequency_cutoff=self.noise_low_freq_cutoff,
                                filter_duration=self.sample_length)
                self.complex_asds[det].append(foo)

    def psd_to_asd(self, psd, start_time, end_time,
                   sample_rate=2048.,
                   low_frequency_cutoff=15.0,
                   filter_duration=128):
        
        psd = psd.copy()

        flen = int(sample_rate / psd.delta_f) // 2 + 1
        oldlen = len(psd)
        psd.resize(flen)

        # Want to avoid zeroes in PSD.
        max_val = psd.max()
        for i in range(len(psd)):
            if i >= (oldlen-1):
                psd.data[i] = psd[oldlen - 2]
            if psd[i] == 0:
                psd.data[i] = max_val

        fil_len = int(filter_duration * sample_rate)
        wn_dur = int(end_time - start_time) + 2 * filter_duration
        if psd.delta_f >= 1. / (2.*filter_duration):
            # If the PSD is short enough, this method is less memory intensive than
            # resizing and then calling inverse_spectrum_truncation
            psd = pycbc.psd.interpolate(psd, 1.0 / (2. * filter_duration))
            # inverse_spectrum_truncation truncates the inverted PSD. To truncate
            # the non-inverted PSD we give it the inverted PSD to truncate and then
            # invert the output.
            psd = 1. / pycbc.psd.inverse_spectrum_truncation(
                                    1./psd,
                                    fil_len,
                                    low_frequency_cutoff=low_frequency_cutoff,
                                    trunc_method='hann')
            psd = psd.astype(complex_same_precision_as(psd))
            # Zero-pad the time-domain PSD to desired length. Zeroes must be added
            # in the middle, so some rolling between a resize is used.
            psd = psd.to_timeseries()
            psd.roll(fil_len)
            psd.resize(int(wn_dur * sample_rate))
            psd.roll(-fil_len)
            # As time series is still mirrored the complex frequency components are
            # 0. But convert to real by using abs as in inverse_spectrum_truncate
            psd = psd.to_frequencyseries()

        kmin = int(low_frequency_cutoff / psd.delta_f)
        psd[:kmin].clear()
        asd = (psd.squared_norm())**0.25
        return asd
    
    def colored_noise(self, asd, start_time, end_time,
                      seed=42, sample_rate=2048.,
                      filter_duration=128, det=None):
        
        """ Create noise from a PSD
    
        Return noise from the chosen PSD. Note that if unique noise is desired
        a unique seed should be provided.
    
        Parameters
        ----------
        asd : pycbc.types.FrequencySeries
            ASD to color the noise
        start_time : int
            Start time in GPS seconds to generate noise
        end_time : int
            End time in GPS seconds to generate noise
        seed : {None, int}
            The seed to generate the noise.
        sample_rate: {16384, float}
            The sample rate of the output data. Keep constant if you want to
            ensure continuity between disjoint time spans.
        filter_duration : {128, float}
            The duration in seconds of the coloring filter
    
        Returns
        --------
        noise : TimeSeries
            A TimeSeries containing gaussian noise colored by the given psd.
        """
        
        white_noise = self.normal(start_time - filter_duration,
                                  end_time + filter_duration,
                                  seed=seed,
                                  sample_rate=sample_rate)

        asd = interpolate(asd, 1.0/(len(white_noise)/2048.))
        white_noise = white_noise.to_frequencyseries()
        # Here we color. Do not want to duplicate memory here though so use '*='
        white_noise *= asd
        colored = white_noise.to_timeseries(delta_t=1.0/sample_rate)
        return colored.time_slice(start_time, end_time)
    
    def normal(self, start, end, sample_rate=2048., seed=0):
        """ Generate data with a white Gaussian (normal) distribution
    
        Parameters
        ----------
        start_time : int
            Start time in GPS seconds to generate noise
        end_time : int
            End time in GPS seconds to generate noise
        sample-rate: float
            Sample rate to generate the data at. Keep constant if you want to
            ensure continuity between disjoint time spans.
        seed : {None, int}
            The seed to generate the noise.
    
        Returns
        --------
        noise : TimeSeries
            A TimeSeries containing gaussian noise
        """

        data = np.random.normal(size=int((end-start)*sample_rate), scale=(sample_rate/2.)**0.5)
        return TimeSeries(data, delta_t=1.0/sample_rate)

    def choose_asd(self):
        # Choose asd for each detector randomly
        # Similar to D3 of MLGWSC-1
        H1_asd = random.choice(self.complex_asds['H1'])
        L1_asd = random.choice(self.complex_asds['L1'])
        return (H1_asd, L1_asd)

    def generate(self, asd, seed, det):
        # Create noise realisation with given ASD
        noise = self.colored_noise(asd,
                                0.0,
                                self.sample_length,
                                seed=seed,
                                sample_rate=self.sample_rate,
                                filter_duration=1.0,
                                det=det)
        noise = noise.numpy()
        return noise

    def apply(self, special, det_only=''):
        # choose a random asd from precomputed set
        time_1 = time.time()
        H1_asd, L1_asd = self.choose_asd()
        # Generate coloured noise using random asd
        rs = np.random.RandomState(seed=special['sample_seed'])
        seeds = list(rs.randint(0, 2**32, 2)) # one for each detector
        H1_noise = self.generate(H1_asd, seeds[0], 'H1')
        L1_noise = self.generate(L1_asd, seeds[1], 'L1')
        noise = np.stack([H1_noise, L1_noise], axis=0)
        return noise


class WhiteNoiseGenerator():
    """ Generate white Gaussian noise for Sage training """
    
    def generate(self, sample_length_in_num, seed=0):
        """ Generate data with a white Gaussian (normal) distribution """
        np.random.seed(seed)
        # 0 mean, 1 std
        return np.random.normal(0, 1, size=sample_length_in_num)

    def apply(self, special, det_only=''):
        # Generate white Gaussian noise using random seeds
        rs = np.random.RandomState(seed=special['sample_seed'])
        seeds = list(rs.randint(0, 2**32, 2)) # one for each detector
        # Get sample length in num
        sample_length_in_s = special['data_cfg'].signal_length # in seconds
        sample_rate = special['data_cfg'].sample_rate # in samples/second
        sample_length_in_num = int(sample_length_in_s * sample_rate)
        # Generate noise for each detector
        H1_noise = self.generate(sample_length_in_num, seeds[0])
        L1_noise = self.generate(sample_length_in_num, seeds[1])
        # Return noise to dataset object
        noise = np.stack([H1_noise, L1_noise], axis=0)
        return noise