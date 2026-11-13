# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Reproducible coloured Gaussian noise of the MLGWSC-1 datasets, adapted
from MLGWSC-1 ``generate_data.py`` and PyCBC
``pycbc.noise.reproduceable``.

Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri Mar 25 13:06:22 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = GPL-3.0-or-later
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

import numpy as np
from numpy.random import RandomState

from pycbc.types import TimeSeries

from sage.core.logger import get_logger

logger = get_logger(__name__)

# This constant need to be constant to be able to recover identical
# results (PyCBC).
BLOCK_SAMPLES = 1638400

class NoiseGenerator(object):
    """
    MLGWSC-1 noise generator: Gaussian noise coloured by an ASD.

    Each detector has its own seed, drawn from ``seed`` with NumPy's
    ``RandomState``; for datasets 2-4 each call draws an ASD per
    detector from ``asds``.

    Parameters
    ----------
    dataset : int
        MLGWSC-1 dataset, 1 to 4.
    seed : int
        Seed of the per-detector seeds and the ASD draws.
    delta_f : float
        Frequency resolution (Hz); the colouring filter lasts
        ``1 / delta_f`` seconds.
    sample_rate : float
        Sample rate (Hz).
    low_frequency_cutoff : float
        Stored; not used.
    detectors : list of str
        Detector names.
    asds : dict or None
        Per detector, the ASDs (``pycbc.types.FrequencySeries``) drawn
        from for datasets 2-4.
    """

    def __init__(self, dataset, seed=42, delta_f=0.04,
                 sample_rate=2048.0, low_frequency_cutoff=15,
                 detectors=['H1', 'L1'], asds=None):
            
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
        """
        Coloured noise of every detector over a GPS interval.

        Parameters
        ----------
        start, end : float
            GPS start and end times (s).
        generate_duration : float or None
            Not used.

        Returns
        -------
        dict of str to pycbc.types.TimeSeries
            Noise per detector.

        Raises
        ------
        RuntimeError
            If the dataset is not 1 to 4.
        """
        # ASD of each detector for the dataset
        keys = {}
        
        if self.dataset == 1:
            logger.debug('Called with dataset 1')
            for det in self.detectors:
                keys[det] = 'aLIGOZeroDetHighPower'
                
        elif self.dataset in [2, 3, 4]:
            logger.debug('Called with dataset {}'.format(self.dataset))
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
        
        """
        Gaussian noise coloured by an ASD (PyCBC ``colored_noise``).

        White noise padded by ``filter_duration`` on each side is
        multiplied by the ASD in the frequency domain, then cut to the
        interval. Unique noise needs a unique seed.

        Parameters
        ----------
        asd : pycbc.types.FrequencySeries
            ASD that colours the noise.
        start_time : int
            Start time in GPS seconds.
        end_time : int
            End time in GPS seconds.
        seed : int
            Seed of the white noise.
        sample_rate : float
            Sample rate of the output. Keep constant to ensure
            continuity between disjoint time spans.
        filter_duration : float
            Duration in seconds of the colouring filter.

        Returns
        -------
        pycbc.types.TimeSeries
            Gaussian noise coloured by the ASD.
        """
        
        white_noise = self.normal(start_time - filter_duration,
                                  end_time + filter_duration,
                                  seed=seed,
                                  sample_rate=sample_rate)
        white_noise = white_noise.to_frequencyseries()
        
        # Here we color. Do not want to duplicate memory here though so
        # use '*='
        white_noise *= asd
        del asd
        colored = white_noise.to_timeseries(delta_t=1.0/sample_rate)
        del white_noise
        return colored.time_slice(start_time, end_time)
    
    
    def normal(self, start, end, sample_rate=2048., seed=0):
        """
        White Gaussian noise over a GPS interval (PyCBC ``normal``).

        Built from fixed-length blocks seeded from ``seed`` and the
        block index, so the same interval always gives the same noise.

        Parameters
        ----------
        start : int
            Start time in GPS seconds.
        end : int
            End time in GPS seconds.
        sample_rate : float
            Sample rate. Keep constant to ensure continuity between
            disjoint time spans.
        seed : int
            Seed of the noise.

        Returns
        -------
        pycbc.types.TimeSeries
            White Gaussian noise.
        """
        
        # This is reproduceable because we used fixed seeds from known
        # values
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
        """
        One block of normal random numbers (PyCBC ``block``).

        Parameters
        ----------
        seed : int
            Seed of the block, taken modulo ``2**32``.
        sample_rate : float
            Sets the variance, ``sample_rate / 2``.

        Returns
        -------
        numpy.ndarray, shape ``(BLOCK_SAMPLES,)``
            Normal random numbers.
        """
        num = BLOCK_SAMPLES
        rng = RandomState(seed % 2**32)
        variance = sample_rate / 2
        return rng.normal(size=num, scale=variance**0.5)
