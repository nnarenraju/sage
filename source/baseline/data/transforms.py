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
import time
import random
import numpy as np
from scipy.signal import butter, sosfiltfilt

# LOCAL
from data.multirate_sampling import multirate_sampling
# from data.parallel_transforms import Parallelise

# PyCBC
from pycbc.psd import inverse_spectrum_truncation
from pycbc.types import TimeSeries

# Parallelisation of transforms
import data.parallel as parallel

# Addressing HDF5 error with file locking (used to address PSD file read error)
# PSD file read has been moved to dataset object. (DEPRECATED)
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"



""" WRAPPERS """

class Unify:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, psds=None, data_cfg=None):
        times = {}
        for transform in self.transforms:
            start = time.time()
            y = transform(y, psds, data_cfg)
            times[transform.__class__.__name__] = time.time() - start
        return y, times


class UnifySignal:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, dets=None, distrs=None, **params):
        times = {}
        for transform in self.transforms:
            start = time.time()
            y = transform(y, dets, distrs, **params)
            times[transform.__class__.__name__] = time.time() - start
        return y, times


class UnifyNoise:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        times = {}
        for transform in self.transforms:
            start = time.time()
            y = transform(y)
            times[transform.__class__.__name__] = time.time() - start
        return y, times


class TransformWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray, psds=None, data_cfg=None):
        if self.always_apply:
            return self.apply(y, psds, data_cfg)
        else:
            pass


class TransformWrapperPerChannel(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply=always_apply)
    
    def __call__(self, y: np.ndarray, psds=None, data_cfg=None):
        channels = y.shape[0]
        # Store transformed array
        augmented = []
        for channel in range(channels):
            if self.always_apply:
                data = y[channel]
                augmented.append(self.apply(data, channel, psds[channel], data_cfg))
            else:
                pass
        return np.stack(augmented, axis=0)


class SignalWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray, dets=None, distrs=None, **params):
        if self.always_apply:
            return self.apply(y, dets, distrs, **params)
        else:
            pass


class NoiseWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            pass
    

#########################################################################################
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
#########################################################################################

class Buffer(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, psds=None, data_cfg=None):
        return y


class Normalise(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, factors=[1.0, 1.0]):
        super().__init__(always_apply)
        assert len(factors) == 2
        self.factors = factors

    def apply(self, y: np.ndarray, channel: int, psd=None, data_cfg=None):
        return y / self.factors[channel]


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
    
    def apply(self, y: np.ndarray, psds=None, data_cfg=None):
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
    
    def apply(self, y: np.ndarray, psds=None, data_cfg=None):
        # Parallelise HighPass filter
        return self.butter_highpass_filter(y)


class Whiten(TransformWrapperPerChannel):
    # PSDs can be different between the channels, so we use perChannel method
    def __init__(self, always_apply=True, trunc_method='hann', remove_corrupted=True):
        super().__init__(always_apply)
        self.trunc_method = trunc_method
        self.remove_corrupted = remove_corrupted
        
    def whiten(self, signal, psd, data_cfg):
        """
        Return a whitened time series

        Parameters
        ----------
        signal : time_series object
            TimeSeries object of the sample
        psd : frequency_series
            PSD used to create the noise_sample
            For dataset 2 & 3, separate file paths for each detector would be given.
            These files are read and passed as frequency_series objects to psd
        max_filter_duration : int
            Maximum length of the time-domain filter in seconds.
        trunc_method : {None, 'hann'}
            Function used for truncating the time-domain filter.
            None produces a hard truncation at `max_filter_len`.
        remove_corrupted : {True, boolean}
            If True, the region of the time series corrupted by the whitening
            is excised before returning. If false, the corrupted regions
            are not excised and the full time series is returned.
        low_frequency_cutoff : {None, float}
            Low frequency cutoff to pass to the inverse spectrum truncation.
            This should be matched to a known low frequency cutoff of the
            data if there is one.

        Returns
        -------
        whitened_data : TimeSeries
            The whitened time series
        
        """
        
        max_filter_len = int(round(data_cfg.whiten_padding * data_cfg.sample_rate))
        
        """ 
        Manipulate PSD for usage in whitening 
        This need not be done (i think) as the psds are created based on signal len anyway
        What would we do when we start using multi-rate sampling?
        """
        # Calculating delta_f of signal and providing that to the PSD interpolation method
        delta_f = data_cfg.delta_f
        # Interpolate the PSD to the required delta_f
        # psd = interpolate(psd, delta_f)
        
        # Interpolate and smooth to the desired corruption length
        psd = inverse_spectrum_truncation(psd,
                                          max_filter_len=max_filter_len,
                                          low_frequency_cutoff=data_cfg.noise_low_freq_cutoff,
                                          trunc_method=self.trunc_method)
        
        """ Whitening """
        # Whiten the data by the asd
        
        # MP mode for whitening
        # pglobal = parallel.SetGlobals(signals, self.process)
        # foo = parallel.Parallelise(pglobal.set_data, pglobal.set_func)
        # foo.args = (delta_f, psd, max_filter_len)
        # foo.name = 'Whitening'
        # white = foo.initiate()
        
        # Sequential mode for whitening
        white = (signal.to_frequencyseries(delta_f=delta_f) / psd**0.5).to_timeseries()
        
        if self.remove_corrupted:
            white = white[int(max_filter_len/2):int(len(white)-max_filter_len/2)]
            
        return white
    
    def process(self, signal, delta_f, psd, max_filter_len):
        white = (signal.to_frequencyseries(delta_f=delta_f) / psd**0.5).to_timeseries()
        return white[int(max_filter_len/2):int(len(white)-max_filter_len/2)]
    
    def apply(self, y: np.ndarray, channel: int, psd=None, data_cfg=None):
        # Convert signal to TimeSeries object
        # --> many signals
        # signals = [TimeSeries(signal, delta_t=1./data_cfg.sample_rate) for signal in y]
        signal = TimeSeries(y, delta_t=1./data_cfg.sample_rate)
        # Whitening
        whitened_sample = self.whiten(signal, psd, data_cfg)
        return whitened_sample


class MultirateSampling(TransformWrapperPerChannel):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, channel: int, psd=None, data_cfg=None):
        # Call multi-rate sampling module for usage
        # This is kept separate since further experimentation might be required
        return multirate_sampling(y, data_cfg)
    


""" Signal only Transformations """

class AugmentPolSky(SignalWrapper):
    """ Used to augment polarisation angle, ra and dec """
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def augment(self, h_plus, h_cross, pol_angle, sky_pos, il, iu, st):
        declination, right_ascension = sky_pos
        # Using PyCBC project_wave to get h_t from h_plus and h_cross
        # Setting the start_time is important! (too late, too early errors are because of this)
        h_plus = TimeSeries(h_plus, delta_t=1./self.sample_rate)
        h_cross = TimeSeries(h_cross, delta_t=1./self.sample_rate)
        # Set start times
        h_plus.start_time = st
        h_cross.start_time = st
        
        # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
        strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle, method='constant') for det in self.dets]
        time_interval = (il, iu)
        # Put both strains together
        return np.array([strain.time_slice(*time_interval, mode='nearest') for strain in strains])
    

    def apply(self, y: np.ndarray, dets=None, distrs=None, **params):
        # Get Augmentation params
        for key, value in params.items():
            setattr(self, key, value)
        
        # Set lal.Detector object as global as workaround for MP methods
        setattr(self, 'dets', dets)
        
        ## Get random value (with a given prior) for polarisation angle, ra, dec
        # Polarisation angle
        # maxlen = len(y[0]) --> many signals
        pol_angles = distrs['pol'].rvs()[0][0]
        # Right ascension, declination
        sky_positions = distrs['sky'].rvs()[0]
        
        times = (self.interval_lower, self.interval_upper, self.start_time, )
        args = (y[0], y[1], pol_angles, sky_positions, ) + times
        out = self.augment(*args)
        
        """
        # Sanity check for sample_length
        for strain in strains:
            to_append = self.sample_length_in_num - len(strain)
            if to_append>0:
                strain.append_zeros(to_append)
            if len(strain) != self.sample_length_in_num:
                raise ValueError("Sample length greater than expected!")
        """
        
        # Input: (h_plus, h_cross) --> output: (det1 h_t, det_2 h_t)
        # Shape remains the same, so reading in dataset object won't be a problem
        return out


class AugmentDistance(SignalWrapper):
    """ Used to augment the distance parameter of the given signal """
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def get_augmented_signal(self, signal, distance, mchirp, distrs):
        distance_old = distance
        # Getting new distance
        chirp_distance = distrs['dchirp'].rvs()[0][0]
        # Producing the new distance with the required priors
        distance_new = chirp_distance * (2.**(-1./5) * 1.4 / mchirp)**(-5./6)
        # Augmenting on the distance
        return (distance_old/distance_new) * signal

    def apply(self, y: np.ndarray, dets=None, distrs=None, **params):
        # TODO: Set all distances during data generation to 1Mpc.
        # Augmenting on distance parameter
        for key, value in params.items():
            setattr(self, key, value)
        
        # Augmentation should be valid if given a batch of signals
        # Run through the augmentation procedure with given dist, mchirp
        augmented_signals = self.get_augmented_signal(y, self.distance, self.mchirp, distrs)
        return augmented_signals



""" Noise only Transformations """

class CyclicShift(NoiseWrapper):
    """ Used to cyclic shift the noise (can be applied to real noise as well) """
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray):
        # Cyclic shifting noise is possible for fake and real noise
        num_roll = random.randint(0, len(y))
        return np.roll(y, num_roll)
