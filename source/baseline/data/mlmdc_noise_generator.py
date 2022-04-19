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
import time
import logging
import numpy as np
from numpy.random import RandomState

# PyCBC handling
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
from pycbc.types import load_frequencyseries, TimeSeries, complex_same_precision_as
from pycbc.psd import interpolate

# This constant need to be constant to be able to recover identical results.
BLOCK_SAMPLES = 1638400

class NoiseGenerator(object):
    
    psd_options = {'H1': [f'../../external/ml-mock-data-challenge-1/psds/H1/psd-{i}.hdf' for i in range(20)],
                   'L1': [f'../../external/ml-mock-data-challenge-1/psds/L1/psd-{i}.hdf' for i in range(20)]}
    
    def __init__(self, dataset, seed=0, delta_f=0.04,
                 sample_rate=2048, low_frequency_cutoff=15,
                 detectors=['H1', 'L1']):
        
        if dataset not in [1, 2, 3]:
            raise ValueError('PsdGenerator is only defined for datasets 1, 2, and 3.')
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.low_frequency_cutoff = low_frequency_cutoff
        self.detectors = detectors
        self.fixed_psds = {det: None for det in self.detectors}
        self.delta_f = delta_f
        self.plen = int(self.sample_rate / self.delta_f) // 2 + 1
        self.rs = np.random.RandomState(seed=seed)
        self.seed = list(self.rs.randint(0, 2**32, len(self.detectors)))
    
    def __call__(self, start, end, generate_duration=3600):
        return self.get(start, end, generate_duration=generate_duration)
    
    def get(self, start, end, generate_duration=3600):        
        keys = {}
        if self.dataset == 1:
            logging.debug('Called with dataset 1')
            for det in self.detectors:
                keys[det] = 'aLIGOZeroDetHighPower'
        elif self.dataset == 2:
            logging.debug('Called with dataset 2')
            for det in self.detectors:
                if self.fixed_psds[det] is None:
                    key = self.rs.randint(0, len(self.psd_options[det]))
                    self.fixed_psds[det] = self.psd_options[det][key]
                keys[det] = self.fixed_psds[det]
        elif self.dataset == 3:
            logging.debug('Called with dataset 3')
            for det in self.detectors:
                key = self.rs.randint(0, len(self.psd_options[det]))
                keys[det] = self.psd_options[det][key]
        else:
            raise RuntimeError(f'Unkown dataset {self.dataset}.')
        
        logging.debug(f'Generated keys {keys}')
        ret = {}
        psds = []
        for i, (det, key) in enumerate(keys.items()):
            logging.debug(f'Starting generating process for detector {det} and key {key}')
            
            #Try loading from frequency series
            psd = load_frequencyseries(key)
            psd = interpolate(psd, self.delta_f) 
            
            if generate_duration is None:
                generate_duration = end - start
                logging.debug('Generate duration was None')
            logging.debug(f'Generate duration set to {generate_duration}')
            done_duration = 0
            noise = None
            #Generate time series noise in chunks
            while done_duration < end - start:
                logging.debug(f'Start of loop with done_duration: {done_duration}')
                segstart = start + done_duration
                segend = min(end, segstart + generate_duration)
                logging.debug(f'Generation segment: {(segstart, segend)} of duration {segend - segstart}')
                
                #Workaround for sample-rate issues
                pad = 0
                duration = segend - segstart + 256
                while 1 / (1 / (duration + pad)) != (duration + pad):
                    pad += 1
                
                tmp = self.colored_noise(psd,
                                         segstart,
                                         segend+pad,
                                         seed=self.seed[i],
                                         sample_rate=self.sample_rate,
                                         low_frequency_cutoff=self.low_frequency_cutoff,
                                         filter_duration=1./self.delta_f)
                tmp = tmp[:len(tmp)-int(pad * tmp.sample_rate)]
                #End of workaround for sample-rate issue
                
                psds.append(psd)

                logging.debug('Succsessfully generated time domain noise')
                if noise is None:
                    logging.debug('Setting noise to tmp')
                    noise = tmp
                else:
                    logging.debug('Appending tmp to noise')
                    noise.append_zeros(len(tmp))
                    noise.data[-len(tmp):] = tmp.data[:]
                done_duration += segend - segstart
                
            logging.debug(f'Exited while loop with done_duration: {done_duration}')
            ret[det] = noise
        
        psds = [np.stack(psd) for psd in psds]
        
        return ret, psds
    
    def colored_noise(self, psd, start_time, end_time,
                  seed=0, sample_rate=16384,
                  low_frequency_cutoff=1.0,
                  filter_duration=128):
        """ Create noise from a PSD
    
        Return noise from the chosen PSD. Note that if unique noise is desired
        a unique seed should be provided.
    
        Parameters
        ----------
        psd : pycbc.types.FrequencySeries
            PSD to color the noise
        start_time : int
            Start time in GPS seconds to generate noise
        end_time : int
            End time in GPS seconds to generate nosie
        seed : {None, int}
            The seed to generate the noise.
        sample_rate: {16384, float}
            The sample rate of the output data. Keep constant if you want to
            ensure continuity between disjoint time spans.
        low_frequency_cutof : {1.0, float}
            The low frequency cutoff to pass to the PSD generation.
        filter_duration : {128, float}
            The duration in seconds of the coloring filter
    
        Returns
        --------
        noise : TimeSeries
            A TimeSeries containing gaussian noise colored by the given psd.
        """
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
        del psd
    
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
    
    def normal(self, start, end, sample_rate=16384, seed=0):
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
