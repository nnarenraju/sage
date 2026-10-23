#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename        : base_classes.py
Description     : Short description of the file

Created on 2026-02-24 16:15:51

__author__        = Narenraju Nagarajan
__copyright__     = Copyright 2026, ProjectName
__license__       = GPL-3.0-or-later
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
    """
    Thin wrapper around a raw data configuration object that adds
    lazily-computed, cached derived quantities.

    :class:`BaseDataConfig` forwards all attribute lookups to the underlying
    ``data_cfg`` object via ``__getattr__``, so any field present on the
    original config is transparently accessible.  Additional temporal and
    frequency-domain quantities that depend on ``sample_rate``,
    ``sample_length_in_s``, and ``padding_length_in_s`` are computed on
    first access and cached via :func:`functools.cached_property`.

    Parameters
    ----------
    data_cfg : object
        Raw data configuration (e.g. a dataclass or namespace) that must
        expose at least ``sample_rate``, ``sample_length_in_s``, and
        ``padding_length_in_s``.

    Cached Properties
    -----------------
    nsamples_in_td : int
        Number of samples in a single analysis window (TD).
    nsamples_in_fd : int
        Number of real-FFT frequency bins for a single window.
    padding_nsamples : int
        Number of padding samples on each edge.
    padded_length_in_s : float
        Total duration including both padding edges (seconds).
    padded_length_in_nsamples : int
        Total sample count including both padding edges.
    padded_delta_f : float
        Frequency resolution of the padded segment (Hz).
    delta_f : float
        Frequency resolution without padding (Hz).
    """

    def __init__(self, data_cfg):
        # Store original
        self.data_cfg = data_cfg

    # Forward attribute access to original config
    def __getattr__(self, name):
        if name == "data_cfg":
            raise AttributeError(name)
        return getattr(self.data_cfg, name)

    # Derived quantities (lazy) cached after one call
    @cached_property
    def nsamples_in_td(self):
        """Number of time-domain samples in the analysis window."""
        return int(self.sample_rate * self.sample_length_in_s)

    @cached_property
    def nsamples_in_fd(self):
        """Number of positive-frequency bins from a real-FFT of the analysis window."""
        return int((self.sample_rate * self.sample_length_in_s) / 2 + 1)

    @cached_property
    def padding_nsamples(self):
        """Number of samples in the one-sided whitening-corruption padding."""
        return int(self.sample_rate * self.padding_length_in_s)

    @cached_property
    def padded_length_in_s(self):
        """Total segment duration including both-sided whitening padding (seconds)."""
        return self.sample_length_in_s + (2.0 * self.padding_length_in_s)

    @cached_property
    def padded_length_in_nsamples(self):
        """Total segment length (samples) including both-sided whitening padding."""
        return int(
            (self.sample_length_in_s + (2.0 * self.padding_length_in_s))
            * self.sample_rate
        )

    @cached_property
    def padded_delta_f(self):
        """Frequency resolution (Hz) of the padded segment."""
        return 1.0 / (self.sample_length_in_s + (2.0 * self.padding_length_in_s))

    @cached_property
    def delta_f(self):
        """Frequency resolution (Hz) of the analysis window (no padding)."""
        return 1.0 / (self.sample_length_in_s)


class BaseConfig:
    """
    Thin wrapper around a raw training/pipeline configuration object.

    Forwards all attribute lookups to the underlying ``cfg`` object via
    ``__getattr__``, providing a uniform interface for pipeline components
    that receive a :class:`BaseConfig` instead of the raw config directly.

    Parameters
    ----------
    cfg : object
        Raw pipeline configuration (e.g. a dataclass or namespace).
    """

    def __init__(self, cfg):
        # Store original
        self.cfg = cfg

    # Forward attribute access to original config
    def __getattr__(self, name):
        if name == "cfg":
            raise AttributeError(name)
        return getattr(self.cfg, name)
