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
import gc
import logging
import os, os.path
import numpy as np

# PyCBC handling
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
from pycbc.types import load_frequencyseries
from pycbc.noise.reproduceable import colored_noise

class NoiseGenerator(object):
    psd_options = {'H1': [f'psds/H1/psd-{i}.hdf' for i in range(20)],
                   'L1': [f'psds/L1/psd-{i}.hdf' for i in range(20)]}
    def __init__(self, dataset, seed=0, filter_duration=128,
                 sample_rate=2048, low_frequency_cutoff=15,
                 detectors=['H1', 'L1']):
        if dataset not in [1, 2, 3]:
            raise ValueError('PsdGenerator is only defined for datasets 1, 2, and 3.')
        self.dataset = dataset
        self.filter_duration = filter_duration
        self.sample_rate = sample_rate
        self.low_frequency_cutoff = low_frequency_cutoff
        self.detectors = detectors
        self.fixed_psds = {det: None for det in self.detectors}
        self.delta_f = 1.0 / self.filter_duration
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
            if isinstance(key, str): #Normal case
                if os.path.isfile(key): #Check if we have to load PSD
                    try:
                        #Try loading from frequency series
                        psd = load_frequencyseries(key)
                    except:
                        #Try loading ASD from txt file
                        psd = pycbc.psd.from_txt(key,
                                                 self.plen,
                                                 self.delta_f,
                                                 self.low_frequency_cutoff,
                                                 is_asd_file=True)
                else:
                    #Try to interpret string as key known to PyCBC
                    logging.debug(f'Now generating PSD from string {key}')
                    psd = pycbc.psd.from_string(key,
                                                self.plen,
                                                self.delta_f,
                                                self.low_frequency_cutoff)
            
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
                
                tmp = colored_noise(psd,
                                    segstart,
                                    segend+pad,
                                    seed=self.seed[i],
                                    sample_rate=self.sample_rate,
                                    low_frequency_cutoff=self.low_frequency_cutoff)
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
                gc.collect()
            logging.debug(f'Exited while loop with done_duration: {done_duration}')
            ret[det] = noise
        
        psds = [np.stack(psd) for psd in psds]
        return ret, psds
