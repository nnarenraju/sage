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

# LOCAL
from data.multirate_sampling import multirate_sampling
from data.snr_calculation import get_network_snr

# PyCBC
import pycbc
from pycbc import DYN_RANGE_FAC
from pycbc.psd import inverse_spectrum_truncation, welch, interpolate
from pycbc.types import TimeSeries, FrequencySeries

# LALSimulation Packages
import lalsimulation as lalsim

# Using segments to read O3a noise
import requests
import ligo.segments

import matplotlib.pyplot as plt

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


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

    def __call__(self, y: np.ndarray, params: dict, special: dict, debug=None):
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
    def __init__(self, generations, fixed, pfixed):
        self.generations = generations
        self.fixed = fixed
        # Probability of selecting a generation or fixed
        self.pfixed = pfixed
    
    def __call__(self, special: dict):
        if special['training']:
            do_fixed = np.random.rand() < self.pfixed
            if do_fixed:
                y = self.fixed.apply()
            else:
                y = self.generations['training'].apply()
        else:
            y = self.generations['validation'].apply()
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
        else:
            pass


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

    def get_cropped(self, y, data_cfg):
        if self.emulate_rmcorrupt:
            self.croplen = int(round(data_cfg.whiten_padding * data_cfg.sample_rate))
        if self.double_sided:
            cropped = y[int(self.croplen/2):int(len(y)-self.croplen/2)]
        else:
            if side == 'left':
                cropped = y[int(self.croplen):]
            elif side == 'right':
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
            ### Estimate the PSD
            raise NotImplementedError('PSD Estimation method not working at the moment!')
            delta_t = 1.0/2048.
            seg_len = int(0.5 / delta_t)
            seg_stride = int(seg_len / 2)
            pure_noise = TimeSeries(pure_noise, delta_t=1./data_cfg.sample_rate)
            psd = welch(pure_noise, seg_len=seg_len, seg_stride=seg_stride)
            psd = interpolate(psd, delta_f)
        
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



""" Waveform Generation """

class FastGenerateWaveform():
    ## Used to augment on all parameters (uses GPU-accelerated IMRPhenomPv2 waveform generation)
    ## Link to Ripple: https://github.com/tedwards2412/ripple
    def __init__(self):
        # Generate the frequency grid (default values)
        self.f_lower = 20.
        self.f_upper = 2048.
        self.delta_f = 0.25
        self.delta_t = 1./2048.
        self.sample_length_in_s = 1./self.delta_f
        self.f_ref = self.f_lower
        # Clean-up params
        self.rwrap = 3.0
        # Tapering params
        beta = 8
        # Condition for optimise f_min
        self.duration_padfactor = 1.1
        # Pick the longest waveform from priors to make some params static
        # Below params are for m1 = 5.01, m2 = 5.0 and with aligned spins s1z, s2z = 0.99
        _theta = {'mass1': 5.01, 'mass2': 5.0, 'spin1z': 0.99, 'spin2z': 0.99}
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
        self.window = np.array(get_window(('kaiser', beta), self.winlen))
        self.kmin = int(self.tmp_f_lower / self.tmp_delta_f)
        self.kmax = self.kmin + self.winlen//2

        # Projection params
        self.signal_length = 12.0 # seconds
        self.whiten_padding = 5.0 # seconds
        self.error_padding_in_s = 0.5 # seconds
    
    def __str__(self):
        data = "f_lower = {}, f_upper = {}, \n \
        delta_f = {}, delta_t = {}, f_ref = {}, \n \
        rwrap = {}".format(self.f_lower, self.f_upper, self.delta_f,
                                   self.delta_t, self.f_ref, self.rwrap)
        return data

    """ ONE-OFF FUNCTIONS (Dont't require JAX or to be jitted in any way) """
    def ms_to_Mc_eta(self, masses):
        ## Converts binary component masses to chirp mass and symmetric mass ratio.
        m1, m2 = masses
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2

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
    
    """ NON-JITTABLES """
    # Using jnp and jitting this function caused significant slowdown
    # Might have something to do with multiple compilations
    def ripple_cleanup(self, hpol):
        # Add the 0th frequency bin back into the fseries
        hpol = np.insert(hpol, 0, 0)
        # ad-hoc high pass filtering
        hpol[self.clean_idx] = 0.0
        return hpol
    
    """ JITTABLES (JAX/JIT implementation removed in Dec 2023) """
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
        # Use jnp to speed things up
        return hpol[0:self.fsize]

    def get_theta_ripple(self, theta):
        # Convert the prior values to jnp array
        # Following params are required for IMRPhenomPv2
        # m1_msun, m2_msun, s1x, s1y, s1z, s2x, s2y, s2z, distance_mpc, tc, phiRef, inclination
        Mc, eta = self.ms_to_Mc_eta(np.array([theta[0], theta[1]]))
        theta_ripple = np.append(np.array([Mc, eta]), np.array(theta[2:]))
        return theta_ripple
    
    def get_theta_pycbc(self, theta):
        # Add required params to waveform kwargs
        theta['f_lower'] = self.tmp_f_lower
        theta['delta_f'] = self.tmp_delta_f
        theta['delta_t'] = self.delta_t
        theta['f_final'] = 2048.0
        return theta
    
    def get_pycbc_hphc(self, theta):
        # Get the IMRPhenomPv2 waveform using PyCBC
        return pycbc.waveform.get_fd_waveform(**theta)

    """ MAIN """
    def make_injection(self, hp, hc, params):
        # Get the required sample length and tc
        sample_length_in_s = len(hp)/2048.
        tc_obs = sample_length_in_s - self.rwrap
        tc_req = params['tc']
        start_time = tc_obs - tc_req
        end_time = tc_obs + (self.signal_length - tc_req)
        # Pad the start and end times for whitening and error padding
        start_time -= (self.whiten_padding/2.0 + self.error_padding_in_s)
        end_time += (self.whiten_padding/2.0 + self.error_padding_in_s)
        # Pad hpol with zeros (append or prepend) if necessary
        left_pad = int(-start_time * 2048.) if start_time < 0.0 else 0
        right_pad = int((end_time-sample_length_in_s) * 2048.) if end_time > sample_length_in_s else 0
        hp = np.pad(hp, (left_pad, right_pad), 'constant', constant_values=(0.0, 0.0))
        hc = np.pad(hc, (left_pad, right_pad), 'constant', constant_values=(0.0, 0.0))
        # Slice the required section out of hpol
        start_idx = int(start_time*2048.) if start_time > 0.0 else 0
        end_idx = int(end_time*2048.) + int(left_pad)
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
        # Tapering and fd_to_td
        hp_td = self.fd_taper_left(hp)
        hc_td = self.fd_taper_left(hc)

        return hp_td, hc_td
    
    def apply(self, params: dict, special: dict):
        # Set lal.Detector object as global as workaround for MP methods
        # Project wave does not work with DataLoader otherwise
        setattr(self, 'dets', special['dets'])
        # Augmentation on all params
        hp, hc = self.generate(params)
        # Make hp, hc into proper injection (adjust to tc and zero pad)
        hp, hc = self.make_injection(hp, hc, params)
        # Convert hp, hc into h(t) using antenna pattern (H1, L1 considered)
        out = self.project(hp, hc, special, params)
        # Input: (h_plus, h_cross) --> output: (det1 h_t, det_2 h_t)
        return out
    


""" Signal only Transformations """


class GenerateWaveform(SignalWrapper):
    ## WARNING: might be too slow, very inefficiently written!
    ## Used to augment on all parameters
    # Produces a new set of h_plus and h_cross arrays
    # Make batch_size priots in dataset object and pass waveform_kwargs one by one
    # Other signal augmentation methods are not required unless it changes the prior distribution.
    def __init__(self, always_apply=True):
        super().__init__(always_apply)
        self.waveform_kwargs = {}
        self.waveform_kwargs['delta_t'] = 1./2048.
        self.waveform_kwargs['f_lower'] = 20.0 # Hz
        self.waveform_kwargs['approximant'] = 'IMRPhenomXPHM'
        self.waveform_kwargs['f_ref'] = 20.0 # Hz

        self.signal_length = 12.0 # seconds 
        self.whiten_padding = 5.0 # seconds
        self.sample_rate = 2048. # Hz
        self.sample_length_in_s = self.signal_length + self.whiten_padding # seconds
        self.sample_length_in_num = round(self.sample_length_in_s * self.sample_rate)
        self.error_padding_in_s = 0.5 # seconds
        self.error_padding_in_num = round(self.error_padding_in_s * self.sample_rate)
        self.signal_low_freq_cutoff = 20.0 # Hz

    def optimise_fmin(self, h_pol):
        # Use self.waveform_kwargs to calculate the fmin for given params
        # Such that the length of the sample is atleast 20s by the time it reaches fmin
        # This DOES NOT mean we produce signals that are exactly 20s long
        current_start_time = -1*h_pol.get_sample_times()[0]
        req_start_time = self.signal_length - h_pol.get_sample_times()[-1]
        fmin = self.signal_low_freq_cutoff*(current_start_time/req_start_time)**(3./8.)
        
        while True:
            # fmin_new is the fmin required for the current params to produce 20.0s signal
            self.waveform_kwargs['f_lower'] = fmin
            h_plus, h_cross = pycbc.waveform.get_td_waveform(**self.waveform_kwargs)
            # Sanity check to verify the new signal length
            new_signal_length = len(h_plus)/self.sample_rate
            if new_signal_length > self.signal_length:
                break
            else:
                fmin = fmin - 3.0
            
        # Return new signal
        return h_plus, h_cross

    def generate(self, prior_values):
        # Convert np.record object to dict and append to waveform_kwargs dict
        self.waveform_kwargs.update(prior_values)
        
        ## Injection
        # Generate the full waveform
        h_plus, h_cross = pycbc.waveform.get_td_waveform(**self.waveform_kwargs)
        # If the signal is smaller than 20s, we change fmin such that it is atleast 20s
        if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] < self.signal_length:
            # Pass h_plus or h_cross
            h_plus, h_cross = self.optimise_fmin(h_plus)
        
        # If it is longer than signal_length, slice out the required region
        if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] > self.signal_length:
            new_end = h_plus.get_sample_times()[-1]
            new_start = -1*(self.signal_length - new_end)
            h_plus = h_plus.time_slice(start=new_start, end=new_end)
            h_cross = h_cross.time_slice(start=new_start, end=new_end)
        
        ## Properly time and project the waveform
        start_time = prior_values['injection_time'] + h_plus.get_sample_times()[0]
        end_time = prior_values['injection_time'] + h_plus.get_sample_times()[-1]
        
        # Calculate the number of zeros to append or prepend
        # Whitening padding will be corrupt and removed in whiten transformation
        start_samp = prior_values['tc'] + (self.whiten_padding/2.0)
        start_interval = prior_values['injection_time'] - start_samp
        # subtract delta value for length error (0.001 if needed)
        end_padding = self.whiten_padding/2.0
        post_merger = self.signal_length - prior_values['tc']
        end_interval = prior_values['injection_time'] + post_merger + end_padding
        
        # Calculate the difference (if any) between two time sets
        diff_start = start_time - start_interval
        diff_end = end_interval - end_time
        # Convert num seconds to num samples
        diff_end_num = int(diff_end * self.sample_rate)
        diff_start_num = int(diff_start * self.sample_rate)
        
        expected_length = ((end_interval-start_interval) + self.error_padding_in_s*2.0) * self.sample_rate
        observed_length = len(h_plus) + (diff_start_num + diff_end_num + self.error_padding_in_num*2.0)
        diff_length = expected_length - observed_length
        if diff_length != 0:
            diff_end_num += diff_length
            
        # If any positive difference exists, add padding on that side
        # Pad h_plus and h_cross with zeros on both end for slicing
        if diff_end > 0.0:
            # Append zeros if we need samples after signal ends
            h_plus.append_zeros(int(diff_end_num + self.error_padding_in_num))
            h_cross.append_zeros(int(diff_end_num + self.error_padding_in_num))
        
        if diff_start > 0.0:
            # Prepend zeros if we need samples before signal begins
            # prepend_zeros arg must be an integer
            h_plus.prepend_zeros(int(diff_start_num + self.error_padding_in_num))
            h_cross.prepend_zeros(int(diff_start_num + self.error_padding_in_num))

        elif diff_start < 0.0:
            h_plus = h_plus.crop(left=-1*((diff_start_num + self.error_padding_in_num)/2048.), right=0.0)
            h_cross = h_cross.crop(left=-1*((diff_start_num + self.error_padding_in_num)/2048.), right=0.0)
        
        assert len(h_plus) == self.sample_length_in_num + self.error_padding_in_num*2.0
        assert len(h_cross) == self.sample_length_in_num + self.error_padding_in_num*2.0
        
        # Setting the start_time, sets epoch and end_time as well within the TS
        # Set the start time of h_plus and h_plus after accounting for prepended zeros
        h_plus.start_time = start_interval - self.error_padding_in_s
        h_cross.start_time = start_interval - self.error_padding_in_s
        # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
        strains = [det.project_wave(h_plus, h_cross, prior_values['ra'], prior_values['dec'], prior_values['polarization'], method='constant') for det in self.dets]
        # Put both strains together
        time_interval = (start_interval, end_interval)
        signal = np.array([strain.time_slice(*time_interval, mode='nearest') for strain in strains])

        return signal

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Sanity check for validation
        if not special['training']:
            return (y, params, special)
        # Set lal.Detector object as global as workaround for MP methods
        # Project wave does not work with DataLoader otherwise
        setattr(self, 'dets', special['dets'])
        # Augmentation on all params
        out = self.generate(params)
        # Input: (h_plus, h_cross) --> output: (det1 h_t, det_2 h_t)
        # Shape remains the same, so reading in dataset object won't be a problem
        return (out, params, special)


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
    def __init__(self, always_apply=True, rescale=True, p=1.0):
        super().__init__(always_apply)
        # If rescale is False, AUG method returns original network_snr, norm_snr and signal
        self.rescale = rescale
        self.p = p

    def _dchirp_from_dist(self, dist, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the chirp distance given the luminosity distance and chirp mass.
        return dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)

    def get_rescaled_signal(self, signal, psds, params, cfg, debug, training, epoch):
        # params: This will contain params var found in __getitem__ method of custom dataset object
        # Get original network SNR
        prelim_network_snr = get_network_snr(signal, psds, params, cfg.export_dir, debug)
        
        if self.rescale:
            # Rescaling the SNR to a uniform distribution within a given range
            rescaled_snr_lower = cfg.rescaled_snr_lower
            rescaled_snr_upper = cfg.rescaled_snr_upper
                
            # Uniform on SNR range
            target_snr = np.random.uniform(rescaled_snr_lower, rescaled_snr_upper)

            rescaling_factor = target_snr/prelim_network_snr
            # Add noise to rescaled signal
            rescaled_signal = signal * rescaling_factor
            
            # Adjust distance parameter for signal according to the new rescaled SNR
            rescaled_distance = params['distance'] / rescaling_factor
            rescaled_dchirp = self._dchirp_from_dist(rescaled_distance, params['mchirp'])
            # Update targets and params with new rescaled distance is not possible
            # We do not know the priors of network_snr properly
            if 'norm_dist' in cfg.parameter_estimation or 'norm_dchirp' in cfg.parameter_estimation:
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
        norms = special['norm']
        epoch = special['epoch']
        # Augmentation on optimal network SNR
        out, params = self.get_rescaled_signal(y, psds, params, cfg, debug, training, epoch)
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
                 shift_psd_along_y=False,
                 use_precomputed=False, h1_psds_hdf="", l1_psds_hdf="",
                 use_generated=False, 
                 use_augmented=False):
        super().__init__(always_apply)
        # Warnings (if required)
        if use_generated:
            warnings.warn('Using generative AI to augment the PSD. Known limitations of supervised learning apply.')
        if use_precomputed:
            warnings.warn('Using precomputed PSDs. Make sure that they do not include PSDs from testing dataset.')
            warnings.warn('Unless using a robust set of PSDs, this method is bound by the limitations of supervised learning')
        
        # Using precomputed (using 81 days of O3a noise)
        # WARNING: This is a cheaty method if this uses testing data PSDs for training.
        self.use_precomputed = use_precomputed
        self.h1_psds = h5py.File(h1_psds_hdf, 'r')
        self.shape_h1_psds = dict(self.h1_psds.attrs)['shape']
        self.l1_psds = h5py.File(l1_psds_hdf, 'r')
        self.shape_l1_psds = dict(self.h1_psds.attrs)['shape']
        # Using generated PSD
        self.use_generated = use_generated
        # Using augmented PSD
        self.use_augmented = use_augmented
        self.shift_psd_along_y = shift_psd_along_y

        # Other params
        self.fs = 2048. #Hz
    
    def estimate_psd(self, y):
        # Compute PSD using Welch's method
        data = [scipy_welch(ts, fs=self.fs, nperseg=4.*self.fs, average='median') for ts in y]
        psds = [datum[1] for datum in data]
        delta_fs = [datum[0][1]-datum[0][0] for datum in data]
        return psds, delta_fs

    def get_psd(self):
        # Use pre-computed PSDs from HDF5 file
        idx_h1 = np.random.randint(0, int(self.shape_h1_psds[0]))
        idx_l1 = np.random.randint(0, int(self.shape_l1_psds[0]))
        new_psds = np.stack([self.h1_psds['data'][idx_h1], self.l1_psds['data'][idx_l1]], axis=0)
        return new_psds

    def generate_psd(self):
        # Generate PSDs using TimeGAN
        # WARNING: Limitations of supervised learning apply
        raise NotImplementedError('generate_psd method under construction!')

    def augment_psd(self):
        # Manually augment PSD
        raise NotImplementedError('augment_psd method under construction!')
    
    def shift_psd(self, new_psds):
        # Shift new PSD along y axis
        raise NotImplementedError('shift_psd method under construction!')
    
    def recolour(self, y, new_psds, old_psds, old_psd_delta_fs):
        ## Whiten the noise using old PSD and recolour using new PSD
        # Convert to frequency domain after windowing
        n = len(y[0])
        delta_t = 1./self.fs
        data_fd = [np.fft.rfft(ts) for ts in y]
        freq = np.fft.rfftfreq(n, delta_t)
        data_delta_f = freq[1] - freq[0]
        # Convert the PSD to new delta_f using PyCBC interpolate function
        old_psds = [FrequencySeries(old_psd, delta_f=delta_f) for old_psd, delta_f in zip(old_psds, old_psd_delta_fs)]
        old_psds = [interpolate(old_psd, data_delta_f) for old_psd in old_psds]
        # Whitening (Remove old PSD from data)
        whitened_signal = [d_fd / np.sqrt(old_psd) for d_fd, old_psd in zip(data_fd, old_psds)]
        # Convert the new PSDs to have delta_f similar to data
        new_psds = [FrequencySeries(new_psd, delta_f=delta_f) for new_psd, delta_f in zip(new_psds, old_psd_delta_fs)]
        new_psds = [interpolate(new_psd, data_delta_f) for new_psd in new_psds]
        # Recolour using new PSD and return to time domain
        recoloured = [np.fft.irfft(white*np.sqrt(new_psd)) for white, new_psd in zip(whitened_signal, new_psds)]
        recoloured = np.stack(recoloured, axis=0)
        return recoloured

    def apply(self, y: np.ndarray, debug=''):
        # Apply given PSD augmentation technique
        old_psds, old_psd_delta_fs = self.estimate_psd(y)
        if self.use_precomputed:
            new_psds = self.get_psd()
        elif self.use_generated:
            new_psds = self.generate_psd()
        elif self.use_augmented:
            new_psds = self.augment_psd(old_psds)
        
        recoloured_noise = self.recolour(y, new_psds, old_psds, old_psd_delta_fs)
        return recoloured_noise



""" Noise Generation """

class GlitchAugmentGWSPY():
    """ Used to augment the noise samples using GWSPY glitch GPS times """
    def __init__(self,
                 H1_O3a_dirname="/local/scratch/igr/nnarenraju/gwspy/H1_O3a_glitches", 
                 L1_O3a_dirname="/local/scratch/igr/nnarenraju/gwspy/L1_O3a_glitches",
                 H1_O3b_dirname="/local/scratch/igr/nnarenraju/gwspy/H1_O3b_glitches", 
                 L1_O3b_dirname="/local/scratch/igr/nnarenraju/gwspy/L1_O3b_glitches"):
        
        # Glitch data files
        self.glitch_files_H1_O3a = [h5py.File(fname) for fname in glob.glob(os.path.join(H1_O3a_dirname, "*.hdf"))]
        # self.num_glitches_H1_O3a = [len(np.array(hf['data'][:])) for hf in self.glitch_files_H1_O3a]
        self.num_glitches_H1_O3a = np.load("./notebooks/tmp/h1_o3a.npy")

        self.glitch_files_L1_O3a = [h5py.File(fname) for fname in glob.glob(os.path.join(L1_O3a_dirname, "*.hdf"))]
        # self.num_glitches_L1_O3a = [len(np.array(hf['data'][:])) for hf in self.glitch_files_L1_O3a]
        self.num_glitches_L1_O3a = np.load("./notebooks/tmp/l1_o3a.npy")

        self.glitch_files_H1_O3b = [h5py.File(fname) for fname in glob.glob(os.path.join(H1_O3b_dirname, "*.hdf"))]
        # self.num_glitches_H1_O3b = [len(np.array(hf['data'][:])) for hf in self.glitch_files_H1_O3b]
        self.num_glitches_H1_O3b = np.load("./notebooks/tmp/h1_o3b.npy")

        self.glitch_files_L1_O3b = [h5py.File(fname) for fname in glob.glob(os.path.join(L1_O3b_dirname, "*.hdf"))]
        # self.num_glitches_L1_O3b = [len(np.array(hf['data'][:])) for hf in self.glitch_files_L1_O3b]
        self.num_glitches_L1_O3b = np.load("./notebooks/tmp/l1_o3b.npy")


    def pick_glitch_file_H1(self, pick_observing_run):
        # Pick a glitch file for each detector
        # Glitch sample obtained from GWOSC using GWSPY GPS times
        H1opt = [self.glitch_files_H1_O3a, self.num_glitches_H1_O3a] if pick_observing_run['H1'] else [self.glitch_files_H1_O3b, self.num_glitches_H1_O3b]
        H1_choice = np.random.randint(low=0, high=len(H1opt[0]))
        H1_file = H1opt[0][H1_choice]
        H1_file_len = H1opt[1][H1_choice]
        return H1_file, H1_file_len
    
    def pick_glitch_file_L1(self, pick_observing_run):
        # Pick a glitch file for each detector
        # Glitch sample obtained from GWOSC using GWSPY GPS times
        L1opt = [self.glitch_files_L1_O3a, self.num_glitches_L1_O3a] if pick_observing_run['L1'] else [self.glitch_files_L1_O3b, self.num_glitches_L1_O3b]
        L1_choice = np.random.randint(low=0, high=len(L1opt[0]))
        L1_file = L1opt[0][L1_choice]
        L1_file_len = L1opt[1][L1_choice]
        return L1_file, L1_file_len

    def read_glitch(self, hf, length):
        # Read requested glitch and return noise sample
        idx = np.random.randint(low=0, high=int(length))
        glitch = np.array(hf['data'][idx]).astype(np.float64)
        return glitch

    def apply(self):
        ## Get random glitch for each detector
        # if randval < 0.5 ('O3a') else ('O3b')
        pick_observing_run = {'H1': np.random.rand() < 0.5, 
                              'L1': np.random.rand() < 0.5}

        # Read the glitch from the provided filenum
        while True:
            H1_file, H1_file_len = self.pick_glitch_file_H1(pick_observing_run)
            glitch_H1 = self.read_glitch(H1_file, H1_file_len)
            if not any(np.isnan(glitch_H1)):
                break
        
        while True:
            L1_file, L1_file_len = self.pick_glitch_file_L1(pick_observing_run)
            glitch_L1 = self.read_glitch(L1_file, L1_file_len)
            if not any(np.isnan(glitch_L1)):
                break
        
        # Augmented noise (Downsampled to 2048. Hz after downloading)
        noise = np.stack([glitch_H1, glitch_L1], axis=0)

        return noise
    

class RandomNoiseSlice():
    """ Used to augment the start time of noise samples from continuous noise .hdf file """
    ### TODO: CHECK FOR BUGS!! ###
    # This will become the primary noise reading function
    def __init__(self, real_noise_path="", sample_length=17.0,
                 segment_llimit=None, segment_ulimit=None):
        self.sample_length = sample_length
        self.min_segment_duration = self.sample_length
        self.real_noise_path = real_noise_path
        self.segment_ends_buffer = 0.0 # seconds
        self.slide_buffer = 240.0
        self.dt = 1./2048.

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
        if segment_ulimit == -1:
            segment_ulimit = len(ligo_segments)

        """ 
        psegment = {}
        for n, seg in enumerate(ligo_segments):
            key_time = str(load_times[seg][0])
            _key = f'{self.detectors[0]}/{key_time}'
            # Sanity check if _key is present in noise file
            try:
                _ = self.O3a_real_noise[_key]
            except:
                # An impossible segment duration and cond rand < segprob is never satisfied
                segdurs[n] = 0
                psegment[n] = [-1, -1, -1]
                continue
            
            # Set valid start and end times of given segment (not actual start time)
            # load_times[seg][0] is the same as seg[0]
            segment_length = len(np.array(self.O3a_real_noise[_key][:]))
            seg_start_idx = 0 + self.segment_ends_buffer
            seg_end_idx = segment_length - (self.sample_length + self.segment_ends_buffer)*(1./self.dt)
            # Get segment duration for calculating sampling ratio wrt all segments
            segdurs[n] = segment_length
            # Add the epoch parameter to store
            psegment[n] = [key_time, seg_start_idx, seg_end_idx]
        
        print(self.real_noise_path)
        """ 
        
        lookup = np.load("./notebooks/tmp/segdurs_all.npy")
        for n, seg in enumerate(ligo_segments):
            key_time = str(load_times[seg][0])
            _key = f'{self.detectors[0]}/{key_time}'
            # Sanity check if _key is present in noise file
            if n >= segment_llimit and n <= segment_ulimit:
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
        seg_idx = np.arange(len(segdurs))
        segprob = list(segdurs/np.sum(segdurs))
        # Get one choice from seg_idx based on probalities obtained from seg durations
        self.segment_choice = lambda _: np.random.choice(seg_idx, 1, p=segprob)[0]

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
    
    def _make_sample_start_time(self, seg_start_idx, seg_end_idx):
        # Make a sample start time that is uniformly distributed within segdur
        return int(np.random.uniform(low=seg_start_idx, high=seg_end_idx))

    def get_noise_segment(self, segdeets):
        ## Get noise sample from given O3a real noise segment
        noise = []
        for det, segdeet in zip(self.detectors, segdeets):
            key_time, seg_start_idx, seg_end_idx = segdeet
            # Get sample_start_time using segment times
            # This start time will lie within a valid segment time interval
            sample_start_idx = self._make_sample_start_time(seg_start_idx, seg_end_idx)
            # Get the required portion of given segment
            sidx = sample_start_idx
            eidx = sample_start_idx + int(self.sample_length / self.dt)
            # Which key does the current segment belong to in real noise file
            # key_time provided is the start time of required segment
            key = f'{det}/{key_time}'
            # Get time series from segment and apply the dynamic range factor
            ts = np.array(self.O3a_real_noise[key][sidx:eidx]).astype(np.float64)
            if "O3a_real_noise.hdf" in self.real_noise_path:
                ts /= DYN_RANGE_FAC
            noise.append(ts)
        
        # Convert noise into np.ndarray, suitable for other transformations
        noise = np.stack(noise, axis=0)
        return noise
    
    def pick_segment(self):
        # Pick a random segment to use based on probablities set using their duration
        # Picking two different segments and start times provides an extra layer of augmentation
        idx1 = self.segment_choice(0)
        idx2 = self.segment_choice(0)
        # Return the segment details of selected segment
        return (self.psegment[idx1], self.psegment[idx2])

    def apply(self):
        ## Get noise sample with random start time from O3a real noise
        # Toss a biased die and retrieve the segment to use
        segdeets = self.pick_segment()
        # Get noise sample with random start time (uniform within segment)
        noise = self.get_noise_segment(segdeets)
        # Return noise data
        return noise
