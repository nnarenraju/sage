# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Mar 25 13:06:22 2022

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

# IN-BUILT
import logging
import numpy as np
from numpy.random import RandomState

# PyCBC handling
from pycbc.types import TimeSeries

# This constant need to be constant to be able to recover identical results.
BLOCK_SAMPLES = 1638400

class NoiseGenerator(object):
    
    def __init__(self, dataset, seed=42, delta_f=0.04,
                 sample_rate=2048.0, low_frequency_cutoff=15,
                 detectors=['H1', 'L1'], asds=None):
        
        if dataset not in [1, 2, 3]:
            raise ValueError('PSDGenerator is only defined for datasets 1, 2, and 3.')
            
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.low_frequency_cutoff = low_frequency_cutoff
        self.detectors = detectors
        self.fixed_asds = {det: None for det in self.detectors}
        self.delta_f = delta_f
        self.plen = int(self.sample_rate / self.delta_f) // 2 + 1
        self.rs = np.random.RandomState(seed=seed)
        self.seed = list(self.rs.randint(0, 2**32, len(self.detectors)))
        self.asd_options = asds
    
    
    def __call__(self, start, end, generate_duration=None):
        return self.get(start, end, generate_duration=generate_duration)
    
    
    def get(self, start, end, generate_duration=None):
        # Get noise PSD data for a given dataset type
        keys = {}
        
        if self.dataset == 1:
            logging.debug('Called with dataset 1')
            for det in self.detectors:
                keys[det] = 'aLIGOZeroDetHighPower'
                
        elif self.dataset == 2:
            logging.debug('Called with dataset 2')
            for det in self.detectors:
                if self.fixed_asds[det] is None:
                    key = self.rs.randint(0, len(self.asd_options[det]))
                    self.fixed_asds[det] = self.asd_options[det][key]
                keys[det] = self.fixed_asds[det]
                
        elif self.dataset == 3:
            logging.debug('Called with dataset 3')
            for det in self.detectors:
                key = self.rs.randint(0, len(self.asd_options[det]))
                keys[det] = self.asd_options[det][key]
        else:
            raise RuntimeError(f'Unkown dataset {self.dataset}.')
        
        ret = {}
        for i, (det, asd) in enumerate(keys.items()):
            tmp = self.colored_noise(asd,
                                     start,
                                     end,
                                     seed=self.seed[i],
                                     sample_rate=self.sample_rate,
                                     filter_duration=1./self.delta_f)
            
            ret[det] = tmp
        
        return ret
    
    
    def colored_noise(self, asd, start_time, end_time,
                  seed=42, sample_rate=2048.,
                  filter_duration=128):
        
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
        white_noise = white_noise.to_frequencyseries()
        
        # Here we color. Do not want to duplicate memory here though so use '*='
        white_noise *= asd
        del asd
        colored = white_noise.to_timeseries(delta_t=1.0/sample_rate)
        del white_noise
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
        
        # This is reproduceable because we used fixed seeds from known values
        block_dur = BLOCK_SAMPLES / sample_rate
        s = int(np.floor(start / block_dur))
        e = int(np.floor(end / block_dur))
    
        # The data evenly divides so the last block would be superfluous
        if end % block_dur == 0:
            e -= 1
    
        sv = RandomState(seed).randint(-2**50, 2**50)
        data = np.concatenate([self.block(i + sv, sample_rate)
                                  for i in np.arange(s, e + 1, 1)])
        ts = TimeSeries(data, delta_t=1.0 / sample_rate, epoch=(s * block_dur))
        return ts.time_slice(start, end)
    
    
    def block(self, seed, sample_rate):
        """ Return block of normal random numbers
    
        Parameters
        ----------
        seed : {None, int}
            The seed to generate the noise.sd
        sample_rate: float
            Sets the variance of the white noise
    
        Returns
        --------
        noise : numpy.ndarray
            Array of random numbers
        """
        num = BLOCK_SAMPLES
        rng = RandomState(seed % 2**32)
        variance = sample_rate / 2
        return rng.normal(size=num, scale=variance**0.5)
