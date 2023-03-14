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
from data.MPB_make_default_dataset import make as make_MPB_default_dataset


# WARNING: Removing any of the parameters present in default will result in errors.

""" DEFAULT """

class Default:
    
    """ Make """
    # if True, a new dataset is created based on the options below
    # else, searches for existing dataset located at os.join(parent_dir, data_dir)
    make_dataset = False
    # Which module to use to create dataset
    # Here, we create a dataset using explicit pycbc functions
    make_module = make_MPB_default_dataset
    
    """ Location (these params used if make_dataset == False, as search loc) """
    # Dataset location directory
    # Data storage drive or /mnt absolute path
    parent_dir = "/local/scratch/igr/nnarenraju"
    # Dataset directory within parent_dir
    data_dir = "dataset_1e6_Jan27_D4_seed42"
    # data_dir = "buffer_datadir"
    
    """ Basic dataset options """
    # These options are used by generate_data.py
    # Type of dataset (1, 2, 3 or 4)
    # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
    dataset = 4
    # Random seed provided to generate_data script
    # This seed is used to generate the priors
    seed = 42

    """ Save Toggle """
    save_injection_priors = True
    
    """ Number of samples """
    # For now, keep both values equal
    num_waveforms = 500000
    num_noises = 500000
    # For efficient RAM usage in data generation
    # Here too, keep both nums equal
    # chunk_size = [num_waveforms_chunk, num_noises_chunk]
    chunk_size = [25000, 25000]
    
    """ Handling number of cores for task """
    # Used in MP and MPB dataset generation methods
    # chunk_size[0] and chunk_size[1] must be divisible exactly by num_queues_datasave
    num_queues_datasave = 1
    num_cores_datagen = 24
    
    """ Save frequency """
    # Save every 'n' number of iterations
    # Set to -1 to never use gc.collect()
    # WARNING!!! - Do NOT use gc.collect when using multiprocessing.
    gc_collect_frequency = -1
    ## this param used if make_dataset == False
    num_sample_save = 10
    
    """ Signal Params """
    ## these params may be used if make_dataset == False
    # Create a new class for a different problem instead of changing this config
    sample_rate = 2048. # Hz
    # (20.0 seconds max + 2.0 seconds of noise padding) would be better
    signal_length = 20.0  # seconds
    # whiten_padding is also known as max_filter_duration in some modules
    whiten_padding = 5.0 # seconds (padding/2.0 on each side of signal_length)
    sample_length_in_s = signal_length + whiten_padding # seconds
    sample_length_in_num = round(sample_length_in_s * sample_rate)
    
    # Error padding (too late/too early errors in time_slice after project_wave)
    # Setting this to 0.1 causes a (PSD, signal) delta_f mismatch error. Annoying.
    error_padding_in_s = 0.5
    error_padding_in_num = round(error_padding_in_s * sample_rate)
    
    signal_low_freq_cutoff = 20.0 # Hz
    signal_approximant = 'IMRPhenomXPHM'
    reference_freq = 20.0 # Hz
    
    prior_low_mass = 7.0 # Msun
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
    
    """ Real O3a noise (Dataset 4) """
    # We probably need to experiment with this a little
    psd_est_segment_length = 36. # in seconds
    psd_est_segment_stride = 18. # in seconds
    
    """ PSD bad band blacking out for Datasets 2,3,4 """
    blackout_max_ratio = 5.0
    
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
    ringdown_leeway = 0.1 # s
    # Seconds before merger to include in max possible sample rate
    merger_leeway=0.2 # s
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
