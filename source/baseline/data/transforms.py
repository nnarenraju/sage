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
import random
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample

# LOCAL
from data.multirate_sampling import multirate_sampling
from data.snr_calculation import get_network_snr
# from data.parallel_transforms import Parallelise

# PyCBC
import pycbc
from pycbc.psd import inverse_spectrum_truncation, welch, interpolate
from pycbc.types import TimeSeries

# Time Stretching
# import librosa
# import pyrubberband

# Addressing HDF5 error with file locking (used to address PSD file read error)
# PSD file read has been moved to dataset object. (DEPRECATED)
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


""" UTILS """

def coin(pof1=0.5):
    return 1 if np.random.random() < pof1 else 0


""" WRAPPERS """

class Unify:
    def __init__(self, transforms: dict):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, pure_noise: np.ndarray, special: dict, key=None):
        transforms = {}
        for transform in self.transforms[key]:
            name = transform.__class__.__name__
            y = transform(y, pure_noise, special)
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


class UnifyNoise:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, debug=None):
        for transform in self.transforms:
            y = transform(y, debug)
        return y


class TransformWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray, pure_noise: np.ndarray, special: dict):
        if self.always_apply:
            return self.apply(y, pure_noise, special)
        else:
            pass


class TransformWrapperPerChannel(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply=always_apply)
    
    def __call__(self, y: np.ndarray, pure_noise: np.ndarray, special: dict):
        channels = y.shape[0]
        # Store transformed array
        augmented = []
        for channel in range(channels):
            if self.always_apply:
                augmented.append(self.apply(y[channel], pure_noise[channel], channel, special))
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
# [4] Multi-rate sampling - Sampling with multiple rates based on GW freq. (0.1s -> 0.02s or less)
# [5] AugmentPolSky - Augmenting on polarisation and sky position. (0.0325s)
# [6] CyclicShift - Time shift noise samples. (5e-5s or less)
# [7] AugmentDistance - Augmenting on GW distance. (0.01s or less)
# 
####################################################################################################

class Buffer(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, pure_noise: np.ndarray, special: dict):
        return y


class Normalise(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, factors=[1.0, 1.0], ignore_factors=False):
        super().__init__(always_apply)
        assert len(factors) == 2
        self.factors = factors
        self.ignore_factors = ignore_factors

    def apply(self, y: np.ndarray, pure_noise:np.ndarray, channel: int, special: dict):
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
    
    def apply(self, y: np.ndarray, pure_noise: np.ndarray, special: dict):
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
    
    def apply(self, y: np.ndarray, pure_noise: np.ndarray, special: dict):
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
    
    def apply(self, y: np.ndarray, pure_noise: np.ndarray, channel: int, special: dict):
        return self.get_cropped(y, special['data_cfg'])


class Whiten(TransformWrapperPerChannel):
    # PSDs can be different between the channels, so we use perChannel method
    def __init__(self, always_apply=True, trunc_method='hann', remove_corrupted=True, estimated=False):
        super().__init__(always_apply)
        self.trunc_method = trunc_method
        self.remove_corrupted = remove_corrupted
        self.estimated = estimated
        self.median_psd = []
        
    def whiten(self, signal, pure_noise, psd, data_cfg):
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
    
    def apply(self, y: np.ndarray, pure_noise: np.ndarray, channel: int, special: dict):
        # Whitening using approximate PSD
        return self.whiten(y, pure_noise, special['psds'][channel], special['data_cfg'])


class MultirateSampling(TransformWrapperPerChannel):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, pure_noise: np.ndarray, channel: int, special: dict):
        # Call multi-rate sampling module for usage
        # This module is kept separate since further experimentation might be required
        return multirate_sampling(y, special['data_cfg'])



""" Signal only Transformations """

class GenerateNewSignal(SignalWrapper):
    ## Used to augment on all parameters (might be too slow)
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
        
        if -1*h_plus.get_sample_times()[0] + h_plus.get_sample_times()[-1] > self.signal_length:
            new_end = h_plus.get_sample_times()[-1]
            new_start = -1*(self.signal_length - new_end)
            h_plus = h_plus.time_slice(start=new_start, end=new_end)
            h_cross = h_cross.time_slice(start=new_start, end=new_end)
        
        ## Properly time and project the waveform (What there is)
        start_time = prior_values['injection_time'] + h_plus.get_sample_times()[0]
        end_time = prior_values['injection_time'] + h_plus.get_sample_times()[-1]
        
        # Calculate the number of zeros to append or prepend (What we need)
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
    """ Used to augment polarisation angle, ra and dec """
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


class AugmentUniformChirpMass(SignalWrapper):
    """ Used to augment the mchirp parameter of the given signal """
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply)

    def get_augmented_signal(self, signal, params, mchirp_lower, mchirp_upper):
        # Get old params
        mchirp_old = params['mchirp']
        # Getting new chirp mass
        mchirp_new = np.random.uniform(mchirp_lower, mchirp_upper, size=1)[0]
        # Get new signal duration from new chirp mass
        lf = 20.0 # Hz
        G = 6.67e-11
        c = 3.0e8 # ms^-1
        tau_old = 5. * (8.*np.pi*lf)**(-8./3.) * (mchirp_old*1.989e30*G/c**3.)**(-5./3.)
        tau_new = 5. * (8.*np.pi*lf)**(-8./3.) * (mchirp_new*1.989e30*G/c**3.)**(-5./3.)
        
        ## Augmenting on the distance
        rate = tau_old/tau_new
        # augmented_signal = [pyrubberband.pyrb.time_stretch(foo, sr=2048, rate=rate) for foo in signal]
        augmented_signal = [librosa.effects.time_stretch(foo, rate=rate) for foo in signal]
        # Time slice the required 20 second portion of the signal using tc param
        tc_in_num_old = int(params['tc']*2048.)
        tc_in_num_new = int((tc_in_num_old)/rate)
        # Add padding on either side before slicing
        augmented_signal = [np.pad(foo, (int(25.0*2048), int(5.0*2048)), 'constant') for foo in augmented_signal]
        # Time slicing while retaining the old time of coalescence
        llimit = (int(25.0*2048)+tc_in_num_new) - (tc_in_num_old + int(2.5*2048))
        ulimit = (int(25.0*2048)+tc_in_num_new) + (int(20.0*2048)-tc_in_num_old) + int(2.5*2048)
        augmented_signal = [foo[llimit:ulimit] for foo in augmented_signal]
        augmented_signal = np.stack(augmented_signal, axis=0)
        # Update params
        params['mchirp'] = mchirp_new
        return (augmented_signal, params)

    def apply(self, y: np.ndarray, params: dict, special: dict, debug=None):
        # Augmenting on distance parameter
        # Unpack required elements from special for augmentation
        mchirp_lower, mchirp_upper = special['limits']['mchirp']
        # Run through the augmentation procedure
        out, params = self.get_augmented_signal(y, params, mchirp_lower, mchirp_upper)
        # Update params
        params.update(params)
        # Update special
        norms = special['norm']
        special['norm_mchirp'] = norms['mchirp'].norm(params['mchirp'])
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
        self.rescale = True
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
