# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Jan 27 00:05:55 2022

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


# LOCAL
from data.make_mlmdc_dataset import make as make_mlmdc_dataset
from data.make_default_dataset import make as make_default_dataset
from data.MP_make_default_dataset import make as make_MP_default_dataset


# WARNING: Removing any of the parameters present in default will result in errors.

""" DEFAULT """

class Default_MLMDC1:
    
    """ Make """
    # if True, a new dataset is created based on the options below
    # else, searches for existing dataset located at os.join(parent_dir, data_dir)
    make_dataset = False
    # Which module to use to create dataset
    # Here, we use the wrapper for the MLMDC1 generate_data.py code
    make_module = make_mlmdc_dataset
    
    """ Location """
    # Dataset location directory
    # Data storage drive or /mnt absolute path
    parent_dir = "/data/wiay/nnarenraju"
    # Dataset directory within parent_dir
    data_dir = "dataset_mlmdc_<size>_20s_D1"
    
    """ Basic dataset options """
    # These options are used by generate_data.py
    # Type of dataset (1, 2, 3 or 4)
    # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
    dataset = 3
    # Random seed provided to generate_data script
    # This will be unique and secret for the testing set
    seed = 42
    # Dataset params
    start_offset = 0
    dataset_duration = 2000
    # Other options
    verbose = True
    force = True
    
    """ pycbc_create_injections options """
    ## make_injections using pycbc_create_injections (uses self.seed)
    # params will be used to call above function via generate_data.py
    # pycbc_create_injections has been modified by nnarenraju (Dec 15th, 2021)
    # Start time of segments
    segment_GPS_start_time = 0.0
    # Time window within which to place the merger
    # 'tc' is located within this window
    time_window_llimit = 14
    time_window_ulimit = 16
    # Length of segment/duration (in seconds)
    segment_length = 20
    # Gap b/w adjacent segments (if any)
    segment_gap = 1
    
    """ Other Options """
    # NOTE: ninjections here is *NOT* provided to pycbc_create_injections
    # Only used to create segments.csv via make_segments.py
    # If 100 injections with 20 second segments are requested,
    # the total duration in segments.csv will be ~2100.0 seconds including gap.
    # This value may be larger than self.duration which is provided to generate_data.py
    # If we request 200s, we obtain the subset of segments required to produce that 
    # from segments.csv. So, we obtain 10 segments, each with one signal.
    ninjections = 100


class Default:
    
    """ Make """
    # if True, a new dataset is created based on the options below
    # else, searches for existing dataset located at os.join(parent_dir, data_dir)
    make_dataset = False
    # Which module to use to create dataset
    # Here, we create a dataset using explicit pycbc functions
    make_module = make_default_dataset
    
    """ Location (these params used if make_dataset == False, as search loc) """
    # Dataset location directory
    # Data storage drive or /mnt absolute path
    parent_dir = "/home/nnarenraju/Research"
    # Dataset directory within parent_dir
    data_dir = "dataset_closest_1e4_20s_D1_checkMP"
    
    """ Basic dataset options """
    # These options are used by generate_data.py
    # Type of dataset (1, 2, 3 or 4)
    # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
    dataset = 1
    # Random seed provided to generate_data script
    # This will be unique and secret for the testing set
    seed = 42

    """ Save Toggle """
    save_injection_priors = True
    
    """ Number of samples """
    # For now, keep both values equal
    num_waveforms = 10000
    num_noises = 10000
    
    """ Save frequency """
    # Save every 'n' number of iterations
    gc_collect_frequency = 1
    ## this param used if make_dataset == False
    sample_save_frequency = 1000
    
    """ Signal Params """
    ## these params may be used if make_dataset == False
    # Create a new class for a different problem instead of changing this config
    sample_rate = 2048. # Hz
    signal_length = 20.0  # seconds
    # whiten_padding is also known as max_filter_duration in some modules
    whiten_padding = 5.0 # seconds (padding/2.0 on each side of signal_length)
    sample_length_in_s = signal_length + whiten_padding # seconds
    sample_length_in_num = round(sample_length_in_s * sample_rate)
    
    signal_low_freq_cutoff = 20.0 # Hz
    signal_approximant = 'IMRPhenomXPHM'
    reference_freq = 20.0 # Hz
    
    prior_low_mass = 10.0 # Msun
    prior_high_mass = 50.0 # Msun
    prior_low_chirp_dist = 130.0
    prior_high_chirp_dist = 350.0
    
    tc_inject_lower = 18.0 # seconds
    tc_inject_upper = 18.2 # seconds
    
    """ PSD Params """
    noise_low_freq_cutoff = 15.0 # Hz
    noise_high_freq_cutoff = 1024.8 # Hz
    delta_f = 1./sample_length_in_s
    # psd_len = round(noise_high_freq_cutoff/delta_f) -> definition depricated
    # Following definition of psd_len taken from:
    # https://pycbc.org/pycbc/latest/html/_modules/pycbc/types/timeseries.html#TimeSeries.to_frequencyseries
    # Got an error in transforms where signal.to_frequencyseries did not have the correct length
    # NOTE: Verified to produce correct results for 1.0 s and 20.0 s signals (March 30th, 2022)
    psd_len = int(int(sample_length_in_num+0.5) / 2 + 1)
    
    """ generate_data.py Noise Params (if used) """
    # Only used in dataset 2 and 3
    # TODO: Not implemented yet. Do not use.
    # if use_example_psd == False
    filter_duration = 128.
    
    """ Multirate sampling params """
    ### Reused params for multi-rate sampling to create bins
    # signal_low_freq_cutoff
    # sample_rate
    # prior_low_mass
    # signal_low_freq_cutoff
    # sample_rate
    # tc_inject_lower
    # tc_inject_upper
    ###
    # Maximum possible signal length in the entire dataset
    # Defined in MLMDC1 as being the longest signal in the testing dataset
    max_signal_length = 20.0 # s
    # Conservative value for max length of ringdown in seconds
    # This ringdown section will be sampled at max possible sample rate
    ringdown_leeway = 0.5 # s
    # Seconds before merger to include in max possible sample rate
    merger_leeway=0.05 # s
    # f_ISCO is multiplied by this factor and used a starting sample freq. at merger
    # If this factor == 2.0, the sampling freq. will be at Nyquist limit
    start_freq_factor=2.5
    # The starting freq. will be reduced by this factor**(n) for n = [0, N], 'N'is num of bins
    # If facrtor==2.0, the sampling freq. is halved every time the signal freq. reduced by
    # a factor equal to fbin_reduction_factor
    fs_reduction_factor=1.9
    # Check value where signal freq. reduces by this factor
    # When this happens, that data idx is store in bins as one of the edges for MR-sampling
    fbin_reduction_factor=2.0
    # Storage bins (DO NOT CHANGE or DELETE)
    dbins = None
