# BUILT-IN
import os
import csv
import glob
import uuid
import h5py
import time
import random
import warnings
import numpy as np

from scipy.signal import butter, sosfiltfilt, resample, get_window
from scipy.signal import welch as scipy_welch
from scipy.signal.windows import tukey
from scipy.stats import beta
from scipy.stats import halfnorm
from numpy.random import RandomState
from scipy.signal import decimate

# LOCAL
from sage.data.transform.multirate_sampling import multirate_sampling
from sage.data.preprocess.snr_calculation import get_network_snr
from sage.data.generation.mlmdc_noise_generator import NoiseGenerator

# PyCBC
import pycbc
from pycbc import DYN_RANGE_FAC
from pycbc.detector import Detector
from pycbc.filter import highpass as pycbc_highpass
from pycbc.psd import inverse_spectrum_truncation, welch, interpolate
from pycbc.types import (
    TimeSeries,
    FrequencySeries,
    load_frequencyseries,
    complex_same_precision_as,
)

# LALSimulation Packages
import lalsimulation as lalsim

# Using segments to read O3a noise
import requests
import ligo.segments

# Plotting
import matplotlib.pyplot as plt

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# This constant need to be constant to be able to recover identical results.
BLOCK_SAMPLES = 1638400
