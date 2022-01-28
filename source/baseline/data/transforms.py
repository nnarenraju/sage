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

import torch
import numpy as np
from scipy.signal import butter, sosfiltfilt


""" WRAPPERS """

class Unify:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y
    

class TransformWrapper:
    def __init__(self, always_apply=False):
        self.always_apply = always_apply

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            pass


class TransformWrapperPerChannel(TransformWrapper):
    def __init__(self, always_apply=False):
        super().__init__(always_apply=always_apply)
    
    def __call__(self, y: np.ndarray):
        channels = y.shape[0]
        if isinstance(y, np.ndarray):
            augmented = y.copy()
        else:
            # If we encounter a pytorch Tensor
            augmented = y.clone()
        for channel in range(channels):
            if self.always_apply:
                augmented[channel] = self.apply(y[channel], channel)
            else:
                pass
        return augmented
    

#########################################################################################
#                             Transforms & their Functionality
# [0] Buffer - absolutely nothing, say it again y'all
# [1] Normalise
# [2] BandPass
#
#
#########################################################################################

class Buffer(TransformWrapper):
    def __init__(self, always_apply=True):
        super().__init__(always_apply)

    def apply(self, y: np.ndarray, channel: int):
        return y


class Normalise(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, factors=[1.0, 1.0]):
        super().__init__(always_apply)
        assert len(factors) == 2
        self.factors = factors

    def apply(self, y: np.ndarray, channel: int):
        return y / self.factors[channel]


class BandPass(TransformWrapperPerChannel):
    def __init__(self, always_apply=True, lower=32, upper=256, fs=2048, order=5):
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
        sos = self.butter_bandpass(self.lower, self.upper, self.fs, order=self.order)
        filtered_data = sosfiltfilt(sos, data)
        return filtered_data
    
    def apply(self, y: np.ndarray, channel: int):
        return self.butter_bandpass_filter(y)
