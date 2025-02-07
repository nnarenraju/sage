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

# LOCAL
from data.MPB_make_default_dataset import make as make_MPB_default_dataset

import numpy as np

# Calculating fudge factor
from pycbc.detector import Detector
from pycbc.conversions import tau_from_final_mass_spin, get_final_from_initial

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lalsimulation as lalsim


# WARNING: Removing any of the parameters present in default will result in errors,
# when running MPB datagen.

""" COMMON """

def get_post_fudge_factor(prior_high_mass):
    """ Post Fudge Factor """
    # Get fudge factor that accounts for wrap around from PyCBC
    # This can be used to estimate the merger+ringdown leeway for MR sampling
    # This should account for waveform content after tc
    m_final, spin_final = get_final_from_initial(mass1=prior_high_mass, 
                                                 mass2=prior_high_mass, 
                                                 spin1z=0.99, spin2z=0.99)
    post_fudge_factor = tau_from_final_mass_spin(m_final, spin_final) * 10 * 1.5 # just in case
    # Adding light travel time between detectors H1 and V1 (We use H1 and L1, but just in case)
    light_travel_time = Detector('H1').light_travel_time_to_detector(Detector('V1')) * 1.1
    post_fudge_factor += light_travel_time
    return post_fudge_factor

def get_imr_chirp_time(m1, m2, s1z, s2z, fl):
    return 1.1 * lalsim.SimIMRPhenomDChirpTime(m1*1.989e+30, m2*1.989e+30, s1z, s2z, fl)


""" DEFAULT """

class Default:

    """ Make """
    OTF = False
    # To handle this: target = 1 if np.random.rand() < self.data_cfg.signal_probability else 0
    signal_probability = 0.5
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
    data_dir = "dataset_D4_1e6_Aug23_vitelotte"
    
    """ Basic dataset options """
    # These options are used by generate_data.py
    # Type of dataset (1, 2, 3 or 4)
    # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
    dataset = 4
    # Random seed provided to generate_data script
    # This seed is used to generate the priors
    seed = 110798

    """ Save Toggle """
    save_injection_priors = True
    
    """ Number of samples """
    # Keep both values equal (balanced dataset)
    # For imbalanced dataset, change the class weights in loss function
    # instead of changing the data generation procedures.
    # Not used for OTF
    num_waveforms = 1_250_000
    num_noises = 1_250_000
    # For efficient RAM usage in data generation
    # Here too, keep both nums equal (Each chunk will be class balanced)
    # chunk_size = [num_waveforms_chunk, num_noises_chunk]
    # sum(chunk_size) must be a divisor of num_waveforms + num_noises
    chunk_size = [25_000, 25_000]

    check_generation = True
    
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
    
    """ Signal Params """
    ## these params may be used if make_dataset == False
    # Create a new class for a different problem instead of changing this config
    sample_rate = 2048. # Hz
    signal_length = 12.0 # seconds
    # whiten_padding is also known as max_filter_duration in some modules
    whiten_padding = 5.0 # seconds (padding/2.0 on each side of signal_length)
    sample_length_in_s = signal_length + whiten_padding # seconds
    sample_length_in_num = round(sample_length_in_s * sample_rate)
    
    # Error padding (combatting too late/too early errors in time_slice after project_wave)
    # Setting this to 0.1 causes a (PSD, signal) delta_f mismatch error. Annoying.
    error_padding_in_s = 0.5
    error_padding_in_num = round(error_padding_in_s * sample_rate)
    
    signal_low_freq_cutoff = 20.0 # Hz
    signal_approximant = 'IMRPhenomPv2'
    reference_freq = 20.0 # Hz

    fix_coin_seeds = False
    fix_signal_seeds = False
    fix_noise_seeds = False
    
    """ PRIORS """
    prior_low_mass = 7.0 # Msun
    prior_high_mass = 50.0 # Msun
    # Chirp distance
    prior_low_chirp_dist = 130.0
    prior_high_chirp_dist = 350.0
    
    tc_inject_lower = 11.0 # seconds
    tc_inject_upper = 11.2 # seconds

    ### MODS ###
    # Modifications to Dataset
    # Possible mods: ('uniform_signal_duration', 'uniform_chirp_mass')
    # NOTE: Set to None if not required
    modification = [None]

    # Both start and end list must sum to 1
    mod_start_probability = [1.0]
    mod_end_probability = [1.0]
    # Annealing is done linear between start and end prob
    # Feature creep: Other functions can be used to move from start to end
    # Annealing is done within the given epoch numbers
    anneal_epochs = [20, 40] # [start, end]
    # Modification off = None option
    modification_toggle_probability = 1.0
    
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
    
    ## To use a mix of real O3a noise and artificial noise created by
    ## colouring Gaussian noise using PSDs estimated from real O3a noise
    # Option used only for dataset == 4
    mixed_noise = False
    mix_ratio = 0.5
    # Use D2/D3 PSDs for D4
    use_d3_psds_for_d4 = True
    
    """ PSD bad band blacking out for Datasets 2,3,4 """
    blackout_max_ratio = 2.0
    
    """ generate_data.py Noise Params (if used) """
    # Only used in dataset 2 and 3
    # TODO: Not implemented yet. Do not use.
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
    srbins_type = 1 # Do not change in Default
    # Maximum possible signal length in the entire dataset
    # Defined in MLMDC1 as being the longest signal in the testing dataset
    max_signal_length = signal_length # s
    # Conservative value for max length of ringdown in seconds
    # This ringdown section will be sampled at max possible sample rate
    ringdown_leeway = 0.1 # s (def: 0.1)
    # Seconds before merger to include in max possible sample rate
    merger_leeway = 0.1 # s (def: 0.1)
    # f_ISCO is multiplied by this factor and used a starting sample freq. at merger
    # If this factor == 2.0, the sampling freq. will be at Nyquist limit
    start_freq_factor = 2.5 # (def: 2.5)
    # The starting freq. will be reduced by this factor**(n) for n = [0, N], 'N'is num of bins
    # If facrtor==2.0, the sampling freq. is halved every time the signal freq. reduced by
    # a factor equal to fbin_reduction_factor
    fs_reduction_factor = 1.8 # (def: 1.8)
    # Check value where signal freq. reduces by this factor
    # When this happens, that data idx is store in bins as one of the edges for MR-sampling
    fbin_reduction_factor = 2.0
    # Removing corrupted samples on either end of MR sampled data
    corrupted_len = 4
    # Storage bins (DO NOT CHANGE or DELETE)
    dbins = None
    network_sample_length = None
    _decimated_bins = None


class DefaultOTF:
    """ Make """
    # Run ORChiD on OTF (on-the-fly data generation) mode
    # Does not try to move extlinks.hdf from dataset dir
    # FIXME: We still need a dataset dir for accessing the PSDs
    OTF = True
    # To handle this: target = 1 if np.random.rand() < self.data_cfg.signal_probability else 0
    signal_probability = 0.5
    
    """ Location (these params used if make_dataset == False, as search loc) """
    # Dataset location directory
    # Data storage drive or /mnt absolute path
    # parent_dir = "/local/scratch/igr/nnarenraju"
    parent_dir = "/home/nnarenraju/Research/ORChiD/"
    # Dataset directory within parent_dir
    # data_dir = "buffer_dataset_check"
    data_dir = "dataset_D4_2e6_Nov28_seed2GW_combination"
    
    """ Basic dataset options """
    # These options are used by generate_data.py
    # Type of dataset (1, 2, 3 or 4)
    # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
    dataset = 4
    # Random seed provided to generate_data script
    # This seed is used to generate the priors
    seed = 110798
    # Fix epoch seeds for lowering dataset variation
    fix_coin_seeds = False
    fix_signal_seeds = False
    fix_noise_seeds = False

    """ OTF Params """
    num_training_samples = 2_000_000
    num_validation_samples = 500_000
    num_auxilliary_samples = 1000
    
    """ Signal Params """
    ## these params may be used if make_dataset == False
    # Create a new class for a different problem instead of changing this config
    sample_rate = 2048. # Hz
    # (20.0 seconds max + 2.0 seconds of noise padding) would be better
    signal_length = 12.0 # seconds
    # Noise padding after ringdown
    # Signal will be placed based on requested noise pad and post fudge factor
    # if signal length is not sufficient for longest possible signal, error occurs.
    noise_pad = 1.5 # seconds
    # whiten_padding is also known as max_filter_duration in some modules
    whiten_padding = 5.0 # seconds (padding/2.0 on each side of signal_length)
    sample_length_in_s = signal_length + whiten_padding # seconds
    sample_length_in_num = round(sample_length_in_s * sample_rate)
    
    # Error padding (combatting too late/too early errors in time_slice after project_wave)
    # Setting this to 0.1 causes a (PSD, signal) delta_f mismatch error. Annoying.
    error_padding_in_s = 0.5
    error_padding_in_num = round(error_padding_in_s * sample_rate)
    
    signal_low_freq_cutoff = 20.0 # Hz
    signal_approximant = 'IMRPhenomPv2'
    reference_freq = 20.0 # Hz

    """ PRIORS """
    prior_low_mass = 7.0 # Msun
    prior_high_mass = 50.0 # Msun
    # Chirp distance
    prior_low_chirp_dist = 130.0
    prior_high_chirp_dist = 350.0

    # Calculate injections time priors
    _longest_wavelen = get_imr_chirp_time(prior_low_mass, prior_low_mass, 0.99, 0.99, signal_low_freq_cutoff)
    post_fudge_factor = get_post_fudge_factor(prior_high_mass)
    # tc params
    tc_diff = 0.2 # seconds
    #tc_inject_lower = signal_length - (noise_pad + post_fudge_factor + tc_diff)
    #tc_inject_upper = tc_inject_lower + tc_diff
    #assert tc_inject_lower > _longest_wavelen, 'longest waveform does not fit within provided signal len!'
    tc_inject_lower = 11.0
    tc_inject_upper = 11.2

    ### MODS ###
    # Modifications to Dataset
    # Possible mods: ('bounded_utau', 'bounded_umc', 'unbounded_utau', 'unbounded_umc', 
    #                 'bounded_plmc', 'bounded_pltau', 'template_placement_metric', 'bounded_umcq',
    #                 'bounded_um1m2')
    # NOTE: Set to None if not required
    modification = [None]
    # modification = [None]
    # Both start and end list must sum to 1
    mod_start_probability = [1.0]
    mod_end_probability = [1.0]
    # Annealing is done linear between start and end prob
    # Feature creep: Other functions can be used to move from start to end
    # Annealing is done within the given epoch numbers
    anneal_epochs = [40, 60] # [start, end]
    # Modification off = None option
    modification_toggle_probability = 1.0
    
    """ Timeslide Analysis """
    # Each detector gets a different signal
    # One detector gets a signal and the other gets noise
    timeslide_mode = False
    tsmode_probability = 0.33
    # Two modes: mode_1=(signal + signal') or mode_2=(signal + noise)
    # This value is used as: 1 if np.random.rand() < p else 2
    # For example: 0.2 --> p=0.2 for mode_1 && p=0.8 for mode_2
    # Set this to 0 or 1 to select one mode or the other
    non_astro_mode_select_probability = 0.5

    """ PSD Params """
    noise_low_freq_cutoff = 15.0 # Hz
    noise_high_freq_cutoff = 1024.8 # Hz
    delta_f = 1./sample_length_in_s
    # psd_len = round(noise_high_freq_cutoff/delta_f) -> definition deprecated
    # Following definition of psd_len taken from:
    # https://pycbc.org/pycbc/latest/html/_modules/pycbc/types/timeseries.html#TimeSeries.to_frequencyseries
    # Got an error in transforms where signal.to_frequencyseries did not have the correct length
    # NOTE: Verified to produce correct results for 1.0 s and 20.0 s signals (March 30th, 2022)
    psd_len = int(int(sample_length_in_num+0.5) / 2 + 1)

    """ Multirate sampling params """
    # Sampling rate bins type 1 or 2
    srbins_type = 1

    ### TYPE 1
    # Maximum possible signal length in the entire dataset
    # Defined in MLMDC1 as being the longest signal in the testing dataset
    max_signal_length = signal_length # s
    # Conservative value for max length of ringdown in seconds
    # This ringdown section will be sampled at max possible sample rate
    ringdown_leeway = 0.1 # s (def: 0.1)
    # Seconds before merger to include in max possible sample rate
    merger_leeway = 0.1 # s (def: 0.1)
    # f_ISCO is multiplied by this factor and used a starting sample freq. at merger
    # If this factor == 2.0, the sampling freq. will be at Nyquist limit
    start_freq_factor = 2.5 # (def: 2.5)
    # The starting freq. will be reduced by this factor**(n) for n = [0, N], 'N'is num of bins
    # If facrtor==2.0, the sampling freq. is halved every time the signal freq. reduced by
    # a factor equal to fbin_reduction_factor
    fs_reduction_factor = 1.8 # (def: 1.8)
    # Check value where signal freq. reduces by this factor
    # When this happens, that data idx is store in bins as one of the edges for MR-sampling
    fbin_reduction_factor = 2.0

    ### TYPE 2
    # These values are to obtain a sample length of 4096 exactly
    # Setting lowest_allowed_fs = 220.0 Hz will return exactly 4096
    # We set this higher to get 4164 and reduce by corrupted length on each side
    # This takes care of any edge effects that might be introduced by decimation
    # All decimation factors should be below 13
    decimation_start_freq = 250 # Hz
    num_blocks = 5
    lowest_allowed_fs = 225 # Hz
    gap_bw_nyquist_and_fs = 42 # Hz
    override_freqs = [20] + [30, 50, 100, 150] + [decimation_start_freq]

    split_with_freqs = False
    split_with_times = True

    # Removing corrupted samples on either end of MR sampled data
    # corrupted_len = [57, 58] for type 2
    corrupted_len = 4
    # Storage bins (DO NOT CHANGE or DELETE)
    dbins = None
    network_sample_length = None
    _decimated_bins = None


class LongerOTF:
    """ Make """
    # Run ORChiD on OTF (on-the-fly data generation) mode
    # Does not try to move extlinks.hdf from dataset dir
    # FIXME: We still need a dataset dir for accessing the PSDs
    OTF = True
    # To handle this: target = 1 if np.random.rand() < self.data_cfg.signal_probability else 0
    signal_probability = 0.5
    
    """ Location (these params used if make_dataset == False, as search loc) """
    # Dataset location directory
    # Data storage drive or /mnt absolute path
    parent_dir = "/local/scratch/igr/nnarenraju"
    # Dataset directory within parent_dir
    # data_dir = "buffer_dataset_check"
    data_dir = "dataset_D4_2e6_Nov28_seed2GW_combination"
    
    """ Basic dataset options """
    # These options are used by generate_data.py
    # Type of dataset (1, 2, 3 or 4)
    # Refer https://github.com/gwastro/ml-mock-data-challenge-1/wiki/Data-Sets 
    dataset = 4
    # Random seed provided to generate_data script
    # This seed is used to generate the priors
    seed = 110798

    """ OTF Params """
    num_training_samples = 2_000_000
    num_validation_samples = 500_000
    num_auxilliary_samples = 125_000
    
    """ Signal Params """
    ## these params may be used if make_dataset == False
    # Create a new class for a different problem instead of changing this config
    sample_rate = 2048. # Hz
    # (20.0 seconds max + 2.0 seconds of noise padding) would be better
    signal_length = 15.0 # seconds
    # Noise padding after ringdown
    # Signal will be placed based on requested noise pad and post fudge factor
    # if signal length is not sufficient for longest possible signal, error occurs.
    noise_pad = 1.5 # seconds
    # whiten_padding is also known as max_filter_duration in some modules
    whiten_padding = 5.0 # seconds (padding/2.0 on each side of signal_length)
    sample_length_in_s = signal_length + whiten_padding # seconds
    sample_length_in_num = round(sample_length_in_s * sample_rate)
    
    # Error padding (combatting too late/too early errors in time_slice after project_wave)
    # Setting this to 0.1 causes a (PSD, signal) delta_f mismatch error. Annoying.
    error_padding_in_s = 0.5
    error_padding_in_num = round(error_padding_in_s * sample_rate)
    
    signal_low_freq_cutoff = 20.0 # Hz
    signal_approximant = 'IMRPhenomXPHM'
    reference_freq = 20.0 # Hz

    """ PRIORS """
    prior_low_mass = 7.0 # Msun
    prior_high_mass = 50.0 # Msun
    # Chirp distance
    prior_low_chirp_dist = 130.0
    prior_high_chirp_dist = 350.0

    # Calculate injections time priors
    _longest_wavelen = get_imr_chirp_time(prior_low_mass, prior_low_mass, 0.99, 0.99, signal_low_freq_cutoff)
    post_fudge_factor = get_post_fudge_factor(prior_high_mass)
    # tc params
    tc_diff = 0.2 # seconds
    tc_inject_lower = signal_length - (noise_pad + post_fudge_factor + tc_diff)
    tc_inject_upper = tc_inject_lower + tc_diff
    assert tc_inject_lower > _longest_wavelen, 'longest waveform does not fit within provided signal len!'

    ### MODS ###
    # Modifications to Dataset
    # Possible mods: ('bounded_utau', 'bounded_umc', 'unbounded_utau', 'unbounded_umc', 'bounded_plmc', bounded_pltau)
    # NOTE: Set to None if not required
    modification = 'bounded_utau'
    modification_probability = 1.0
    
    """ PSD Params """
    noise_low_freq_cutoff = 15.0 # Hz
    noise_high_freq_cutoff = 1024.8 # Hz
    delta_f = 1./sample_length_in_s
    # psd_len = round(noise_high_freq_cutoff/delta_f) -> definition deprecated
    # Following definition of psd_len taken from:
    # https://pycbc.org/pycbc/latest/html/_modules/pycbc/types/timeseries.html#TimeSeries.to_frequencyseries
    # Got an error in transforms where signal.to_frequencyseries did not have the correct length
    # NOTE: Verified to produce correct results for 1.0 s and 20.0 s signals (March 30th, 2022)
    psd_len = int(int(sample_length_in_num+0.5) / 2 + 1)
    
    """ Multirate sampling params """
    # Sampling rate bins type 1 or 2
    srbins_type = 1

    ### TYPE 1
    # Maximum possible signal length in the entire dataset
    # Defined in MLMDC1 as being the longest signal in the testing dataset
    max_signal_length = signal_length # s
    # Conservative value for max length of ringdown in seconds
    # This ringdown section will be sampled at max possible sample rate
    ringdown_leeway = 0.1 # s (def: 0.1)
    # Seconds before merger to include in max possible sample rate
    merger_leeway = 0.1 # s (def: 0.1)
    # f_ISCO is multiplied by this factor and used a starting sample freq. at merger
    # If this factor == 2.0, the sampling freq. will be at Nyquist limit
    start_freq_factor = 2.5 # (def: 2.5)
    # The starting freq. will be reduced by this factor**(n) for n = [0, N], 'N'is num of bins
    # If facrtor==2.0, the sampling freq. is halved every time the signal freq. reduced by
    # a factor equal to fbin_reduction_factor
    fs_reduction_factor = 1.8 # (def: 1.8)
    # Check value where signal freq. reduces by this factor
    # When this happens, that data idx is store in bins as one of the edges for MR-sampling
    fbin_reduction_factor = 2.0

    ### TYPE 2
    # These values are to obtain a sample length of 4096 exactly
    # Setting lowest_allowed_fs = 220.0 Hz will return exactly 4096
    # We set this higher to get 4164 and reduce by corrupted length on each side
    # This takes care of any edge effects that might be introduced by decimation
    # All decimation factors should be below 13
    decimation_start_freq = 250 # Hz
    num_blocks = 5
    lowest_allowed_fs = 225 # Hz
    gap_bw_nyquist_and_fs = 42 # Hz
    override_freqs = [20] + [30, 50, 100, 150] + [decimation_start_freq]

    split_with_freqs = False
    split_with_times = True

    # Removing corrupted samples on either end of MR sampled data
    # corrupted_len = [57, 58] for type 2
    corrupted_len = 4
    # Storage bins (DO NOT CHANGE or DELETE)
    dbins = None
    network_sample_length = None
    _decimated_bins = None


class Legacy:
    
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
    parent_dir = "/home/nnarenraju/Research/ORChiD/"
    # Dataset directory within parent_dir
    data_dir = "dataset_D4_2e6_Nov28_seed2GW_combination"
    
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
    # Sampling rate bins type 1 or 2
    srbins_type = 1
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
    corrupted_len = 4
    # Storage bins (DO NOT CHANGE or DELETE)
    dbins = None    