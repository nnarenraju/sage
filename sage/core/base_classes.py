#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : base_classes.py
Description     : Short description of the file

Created on 2026-02-24 16:15:51

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = MIT Licence
__version__       = 0.0.1
__maintainer__    = Narenraju Nagarajan
__affiliation__   = N/A
__email__         = N/A
__status__        = ['inProgress', 'Archived', 'inUsage', 'Debugging']


GitHub Repository: NULL

Documentation: NULL

"""

# Packages
import torch
import numpy as np

from functools import cached_property


class BaseDataConfig:

    def __init__(self, data_cfg):
        # Store original
        self.data_cfg = data_cfg

    # Forward attribute access to original config
    def __getattr__(self, name):
        return getattr(self.data_cfg, name)

    # Derived quantities (lazy) cached after one call
    @cached_property
    def nsamples_in_td(self):
        return int(self.sample_rate * self.sample_length_in_s)

    @cached_property
    def nsamples_in_fd(self):
        return int((self.sample_rate * self.sample_length_in_s) / 2 + 1)

    @cached_property
    def padding_nsamples(self):
        return int(self.sample_rate * self.padding_length_in_s)

    @cached_property
    def padded_length_in_s(self):
        return self.sample_length_in_s + (2.0 * self.padding_length_in_s)

    @cached_property
    def padded_length_in_nsamples(self):
        return int(
            (self.sample_length_in_s + (2.0 * self.padding_length_in_s))
            * self.sample_rate
        )

    @cached_property
    def padded_delta_f(self):
        return 1.0 / (self.sample_length_in_s + (2.0 * self.padding_length_in_s))

    @cached_property
    def delta_f(self):
        return 1.0 / (self.sample_length_in_s)


class BaseConfig:

    def __init__(self, cfg):
        # Store original
        self.cfg = cfg

    # Forward attribute access to original config
    def __getattr__(self, name):
        return getattr(self.cfg, name)
