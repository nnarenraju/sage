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
import h5py
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# LOCAL
from data.multirate_sampling import multirate_sampling

# PyCBC
import pycbc
from pycbc import distributions
from pycbc.psd import inverse_spectrum_truncation, interpolate
from pycbc.types import load_frequencyseries, TimeSeries, FrequencySeries

# Addressing HDF5 error with file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


""" WRAPPERS """

class Unify:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, psds=None, data_cfg=None):
        for transform in self.transforms:
            y = transform(y, psds, data_cfg)
        return y


class UnifySignalOnly:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, dets=None, time_interval=None, distrs=None):
        for transform in self.transforms:
            y = transform(y, dets, time_interval, distrs)
        return y
    

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
                augmented.append(self.apply(y[channel], channel, psds[channel], data_cfg))
            else:
                pass
        return np.array(augmented, dtype=np.float64)


class ProjectionWrapper:
    def __init__(self, always_apply=True):
        self.always_apply = always_apply
    
    def __call__(self, y: np.ndarray, dets=None, time_interval=None, distrs=None):
        if self.always_apply:
            return self.apply(y, dets, time_interval, distrs)
        else:
            pass
    

#########################################################################################
#                             Transforms & their Functionality
# [0] Buffer - Absolutely nothing, say it again y'all
# [1] Normalise - Normalisation of each sample wrt entire dataset
# [2] BandPass - Butter bandpass filter. Uses sosfiltfilt function for stability.
# [3] Whitening - PyCBC whitening function. PSD input required to whiten.
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
        sos = butter(self.order, [low, high], analog=False, btype='band', output='sos')
        return sos

    def butter_bandpass_filter(self, data):
        sos = self.butter_bandpass()
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data
    
    def apply(self, y: np.ndarray, psds=None, data_cfg=None):
        # Verified to produce the same results as PerChannel mode
        return self.butter_bandpass_filter(y)


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
        # psd1 = interpolate(psd, delta_f)
        
        # Interpolate and smooth to the desired corruption length
        psd = inverse_spectrum_truncation(psd,
                                          max_filter_len=max_filter_len,
                                          low_frequency_cutoff=data_cfg.noise_low_freq_cutoff,
                                          trunc_method=self.trunc_method)
        
        """ Whitening """
        # Whiten the data by the asd
        white = (signal.to_frequencyseries(delta_f=delta_f) / psd**0.5).to_timeseries()

        if self.remove_corrupted:
            white = white[int(max_filter_len/2):int(len(signal)-max_filter_len/2)]
            
        return white
        
    def apply(self, y: np.ndarray, channel: int, psd=None, data_cfg=None):
        # Convert signal to TimeSeries object
        signal = TimeSeries(y, delta_t=1./data_cfg.sample_rate)
        # Whitening
        whitened_sample = self.whiten(signal, psd, data_cfg)
        return whitened_sample


class MultirateSampling(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, psds=None, data_cfg=None):
        # Call multi-rate sampling module for usage
        # This is kept separate since further experimentation might be required
        return multirate_sampling(y, data_cfg)
    

class AugmentDistance(TransformWrapper):
    """ Used to augment the distance parameter of the given signal """
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, psds=None, data_cfg=None):
        # Augmenting on distance parameter
        pass
    
    

""" Signal only Transformations """

class ProjectWave(ProjectionWrapper):
    """ Used to augment polarisation angle, ra and dec """
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, dets=None, time_interval=None, distrs=None):
        
        # Using PyCBC project_wave to get h_t from h_plus and h_cross
        # TODO: h_plus and h_cross have to be TimeSeries objects when saved
        h_plus, h_cross = y[0], y[1]
        ## Get random value (with a given prior) for polarisation angle, ra, dec
        # Polarisation angle
        uniform_angles = distrs['pol'].rvs(size=1)
        pol_angle = uniform_angles[0][0]
        # Right ascension, declination
        declination, right_ascension = distrs['sky'].rvs()[0]
        # Use project_wave and random realisation of polarisation angle, ra, dec to obtain augmented signal
        strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle) for det in dets]
        strains = [strain.time_slice(*time_interval) for strain in strains]
        
        # Sanity check for sample_length
        for strain in strains:
            to_append = self.sample_length_in_num - len(strain)
            if to_append>0:
                strain.append_zeros(to_append)
            if len(strain) != self.sample_length_in_num:
                raise ValueError("Sample length greater than expected!")
        
        # Input: (h_plus, h_cross) --> output: (det1 h_t, det_2 h_t)
        # Shape remains the same, so reading in dataset object won't be a problem
        return strains
