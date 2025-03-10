# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Mon Dec  6 11:07:42 2021

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
import os
import re
import h5py
import glob
import torch
import random
import numpy as np
import configparser

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


# LOCAL
from data.psd_loader import load_psds
from data.prior_modifications import *
from data.distributions import get_distributions
from data.normalisation import get_normalisations
from data.snr_calculation import get_network_snr
from data.plot_dataloader_unit import plot_unit
from data.multirate_sampling import get_sampling_rate_bins_type1, get_sampling_rate_bins_type2
from data.pycbc_draw_samples import read_config

# PyCBC
import pycbc
from pycbc import transforms
from pycbc import distributions
from pycbc.types import TimeSeries
from pycbc.conversions import det_tc
from pycbc.types import FrequencySeries
from pycbc.psd import welch, interpolate
from pycbc.distributions.power_law import UniformRadius
from pycbc.distributions.utils import draw_samples_from_config

# Datatype for storage
tensor_dtype=torch.float32



""" Utility Classes """

class Normalise:
    """
    Normalise the parameter using prior ranges
    
        For example, norm_tc = (tc - min_val)/(max_val - min_val)
        The values of max_val and min_val are provided
        to the class. self.get_norm can be called during
        data generation to get normalised values of tc, if needed.
    
    """
    
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def norm(self, val):
        # Return lambda to use for normalisation
        return (val - self.min_val)/(self.max_val - self.min_val)


""" Dataset Objects """

class MinimalOTF(Dataset):
    # NOTE: Currently, this works only for Dataset 4
    
    def __init__(self, data_paths=None, targets=None,
                 waveform_generation=None, noise_generation=None,
                 transforms=None, target_transforms=None,
                 signal_only_transforms=None, noise_only_transforms=None, 
                 training=False, aux=False, store_device='cpu', train_device='cpu', 
                 cfg=None, data_cfg=None):
        
        super().__init__()
        # Non-OTF params
        self.data_paths = data_paths
        self.targets = targets
        # Common params
        self.waveform_generation = waveform_generation
        self.noise_generation = noise_generation
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.signal_only_transforms = signal_only_transforms
        self.noise_only_transforms = noise_only_transforms
        self.store_device = store_device
        self.train_device = train_device
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.fix_signal_seeds = self.data_cfg.fix_signal_seeds
        self.fix_noise_seeds = self.data_cfg.fix_noise_seeds
        # Using this for now to get psds (TODO: move psd creation from datagen)
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        self.epoch = -1
        self.cflag = -1
        self.aux = aux

        if self.data_cfg.OTF:
            if training:
                self.total_samples_per_epoch = self.data_cfg.num_training_samples
            else:
                if not self.aux:
                    self.total_samples_per_epoch = self.data_cfg.num_validation_samples
                else:
                    self.total_samples_per_epoch = self.data_cfg.num_auxilliary_samples
        else:
            if training:
                self.total_samples_per_epoch = 0.8 * (self.data_cfg.num_waveforms + self.data_cfg.num_noises)
            else:
                if not self.aux:
                    self.total_samples_per_epoch = 0.2 * (self.data_cfg.num_waveforms + self.data_cfg.num_noises)
                else:
                    self.total_samples_per_epoch = 0.01 * (self.data_cfg.num_waveforms + self.data_cfg.num_noises)
        
        self.training = training
        
        # Set CUDA device for pin_memory if needed
        if isinstance(self.cfg.store_device, torch.device):
            # setattr(self, 'foo', torch.cuda.set_device(self.cfg.store_device))
            setattr(self, 'foo', self.cfg.store_device)
        
        """ Set default waveform generation params """
        class_exists = lambda clist, cname: any([foo for foo in clist if foo.__class__.__name__==cname])
        get_class = lambda clist, cname: [foo for foo in clist if foo.__class__.__name__==cname][0]

        if self.cfg.generation['signal'] != None:
            # Use data_cfg to set waveform generation class attributes
            if class_exists(self.waveform_generation.generations, 'FastGenerateWaveform'):
                wgen = get_class(self.waveform_generation.generations, 'FastGenerateWaveform')
                wgen.f_lower = data_cfg.signal_low_freq_cutoff
                wgen.f_upper = data_cfg.sample_rate
                wgen.delta_t = 1./data_cfg.sample_rate
                wgen.f_ref = data_cfg.reference_freq
                wgen.signal_length = data_cfg.signal_length
                wgen.whiten_padding = data_cfg.whiten_padding
                wgen.error_padding_in_s = data_cfg.error_padding_in_s
                wgen.sample_rate = data_cfg.sample_rate
                # Precompute common params for waveform generation
                wgen.precompute_common_params()

        """ Set default noise generation params """
        if self.cfg.generation['noise'] != None:
            for name in ['training', 'validation']:
                noigen_name = self.noise_generation.generations[name].__class__.__name__
                self.noise_generation.generations[name].sample_length = data_cfg.signal_length + data_cfg.whiten_padding # seconds
                if noigen_name == 'RandomNoiseSlice':
                    self.noise_generation.generations[name].dt = 1./data_cfg.sample_rate # seconds
                elif noigen_name == 'ColouredNoiseGenerator':
                    self.noise_generation.generations[name].delta_f = self.data_cfg.delta_f
                    self.noise_generation.generations[name].noise_low_freq_cutoff = self.data_cfg.noise_low_freq_cutoff
                    self.noise_generation.generations[name].sample_rate = self.data_cfg.sample_rate
                # Compute common params
                self.noise_generation.generations[name].precompute_common_params()

            if self.noise_generation.aux != None:
                self.noise_generation.aux.sample_length = data_cfg.signal_length + data_cfg.whiten_padding # seconds

        """ Set default recolour transformation params """
        if self.noise_only_transforms != None:
            recolour = get_class(self.noise_only_transforms.transforms, 'Recolour')
            if recolour != []:
                recolour.fs = data_cfg.sample_rate # Hz
                recolour.sample_length_in_s = data_cfg.signal_length + data_cfg.whiten_padding # seconds
                recolour.noise_low_freq_cutoff = data_cfg.noise_low_freq_cutoff # Hz
                recolour.signal_low_freq_cutoff = data_cfg.signal_low_freq_cutoff # Hz
                recolour.whiten_padding = data_cfg.whiten_padding # seconds

        """ PSD Handling (used in whitening) """
        # Store the PSD files here in RAM. This reduces the overhead when whitening.
        # Read all psds in the data_dir and store then as FrequencySeries
        self.psds_data = load_psds(self.data_loc, self.data_cfg)
            
        """ Multi-rate Sampling """
        # Get the sampling rates and their bins idx
        if self.data_cfg.srbins_type == 1:
            self.data_cfg.dbins = get_sampling_rate_bins_type1(self.data_cfg)
        elif self.data_cfg.srbins_type == 2:
            self.data_cfg.dbins = get_sampling_rate_bins_type2(self.data_cfg)
        
        """ LAL Detector Objects (used in project_wave - AugPolSky) """
        # Detector objects (these are lal objects and may present problems when parallelising)
        # Create the detectors (TODO: generalise this!!!)
        self.detectors_abbr = ('H1', 'L1')
        self.detectors = [pycbc.detector.Detector(det_abbr) for det_abbr in self.detectors_abbr]
        
        """ PyCBC Distributions (used in AugPolSky and AugDistance) """
        ## Distribution objects for augmentation
        self.distrs = get_distributions(self.data_cfg)

        """ Normalising the augmented params (if needed) """
        self.norm, self.limits = get_normalisations(self.cfg, self.data_cfg)

        """ Useful params """
        self.sample_rate = self.data_cfg.sample_rate
        self.noise_low_freq_cutoff = self.data_cfg.noise_low_freq_cutoff

        if not self.data_cfg.OTF:
            """ Random noise realisation """
            self.noise_idx = np.argwhere(self.targets == 0).flatten()
            
            """ Keep ExternalLink Lookup table open till end of run """
            lookup = os.path.join(cfg.export_dir, 'extlinks.hdf')
            self.extmain = h5py.File(lookup, 'r', libver='latest')

        ## SPECIAL
        self.special = {}
        self.special['distrs'] = self.distrs
        self.special['norm'] = self.norm
        self.special['cfg'] = self.cfg
        self.special['data_cfg'] = self.data_cfg
        self.special['dets'] = self.detectors
        self.special['psds'] = self.psds_data
        self.special['training'] = self.training
        self.special['aux'] = self.cflag
        self.special['limits'] = self.limits
        self.special['fix_signal_seeds'] = self.fix_signal_seeds
        self.special['fix_noise_seeds'] = self.fix_noise_seeds
        self.special['default_keys'] = self.special.keys()

        ## Ignore Params
        self.ignore_params = {'start_time', 'interval_lower', 'interval_upper', 
                              'sample_rate', 'noise_low_freq_cutoff', 'declination', 
                              'right_ascension', 'polarisation_angle'}
        
        """ Waveform Parameters """
        ## Parameters required to produce a waveform
        # m1_msun, m2_msun, s1x, s1y, s1z, s2x, s2y, s2z, distance_mpc, tc, phiRef, inclination
        self.params = {'mass1': -1, 'mass2': -1, 'spin1x': -1, 'spin1y': -1, 'spin1z': -1,
                       'spin2x': -1, 'spin2y': -1, 'spin2z': -1, 'distance': -1, 'tc': -1,
                       'coa_phase': -1,  'inclination': -1}
        
        ini_parent = '/home/nnarenraju/Research/sgwc-1/sage/ini_files'
        dataset = 4 # MinimalOTF made specifically for dataset 4
        CONFIG_PATH = "{}/ds{}.ini".format(ini_parent, dataset)
        self.randomsampler, self.waveform_transforms = read_config(path=CONFIG_PATH)
        # data_config and ini_file both have definitions of tc range
        # There may be a discrepancy between the two
        config_reader = configparser.ConfigParser()
        config_reader.read(CONFIG_PATH)
        assert float(config_reader['prior-tc']['min-tc']) == self.data_cfg.tc_inject_lower, 'min-tc discrepancy in ini file'
        assert float(config_reader['prior-tc']['max-tc']) == self.data_cfg.tc_inject_upper, 'max-tc discrepancy in ini file'
        # mass check
        assert float(config_reader['prior-mass1']['min-mass1']) == self.data_cfg.prior_low_mass, 'min-m1 discrepancy in ini file'
        assert float(config_reader['prior-mass1']['max-mass1']) == self.data_cfg.prior_high_mass, 'max-m1 discrepancy in ini file'
        assert float(config_reader['prior-mass2']['min-mass2']) == self.data_cfg.prior_low_mass, 'min-m2 discrepancy in ini file'
        assert float(config_reader['prior-mass2']['max-mass2']) == self.data_cfg.prior_high_mass, 'max-m2 discrepancy in ini file'
        # dchirp check
        assert float(config_reader['prior-chirp_distance']['min-chirp_distance']) == self.data_cfg.prior_low_chirp_dist, 'min-dchirp discrepancy in ini file'
        assert float(config_reader['prior-chirp_distance']['max-chirp_distance']) == self.data_cfg.prior_high_chirp_dist, 'max-dchirp discrepancy in ini file'

        """ Prior Modifications """
        self.bprior = BoundedPriors(self.data_cfg.prior_low_mass, self.data_cfg.prior_high_mass, self.data_cfg.signal_low_freq_cutoff)
        # Set modification probabilities for epochs
        _modprobs = zip(self.data_cfg.mod_start_probability, self.data_cfg.mod_end_probability)
        self.modprobs = {}
        start_epoch, end_epoch = self.data_cfg.anneal_epochs
        anneal_epochs = np.arange(start_epoch, end_epoch, 1)
        for mod_method, probs in zip(data_cfg.modification, _modprobs):
            nsplit = end_epoch - start_epoch
            all_probs = np.linspace(probs[0], probs[1], nsplit)
            self.modprobs[mod_method] = {epoch:prob for epoch, prob in zip(anneal_epochs, all_probs)}
            # Add padding probs if needed
            leftover_epochs_before = np.arange(0, start_epoch, 1)
            before_update = {epoch:probs[0] for epoch in leftover_epochs_before}
            leftover_epochs_after = np.arange(end_epoch, cfg.num_epochs, 1)
            after_update = {epoch:probs[1] for epoch in leftover_epochs_after}
            # Update modprobs with the leftover epochs
            self.modprobs[mod_method].update(before_update)
            self.modprobs[mod_method].update(after_update)
        
        # Plotting modification probabilites for all epochs
        if self.training:
            save_modprobs = os.path.join(cfg.export_dir, 'prior_modification_probs.png')
            plt.figure(figsize=(9.0, 9.0))
            for method, probdata in self.modprobs.items():
                plot_epochs = list(probdata.keys())
                plot_probs = list(probdata.values())
                foo = np.column_stack((plot_epochs, plot_probs))
                foo = foo[foo[:,0].argsort()]
                method = 'U(m1, m2)' if method == None else method
                plt.plot(foo[:,0], foo[:,1], label=method, linewidth=3.0)
            plt.xlabel('Number of Epochs')
            plt.ylabel('Probability of Choosing')
            plt.xlim(0, 100) # limit num epochs
            plt.legend()
            plt.savefig(save_modprobs)

        # Temporary Messages
        # print('WARNING: ')

    def __len__(self):
        if self.data_cfg.OTF:
            return self.total_samples_per_epoch
        else:
            return len(self.data_paths)
    

    def _dchirp_from_dist(self, dist, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the chirp distance given the luminosity distance and chirp mass.
        return dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)


    def _dist_from_dchirp(self, chirp_distance, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the luminosity distance given a chirp distance and chirp mass.
        return chirp_distance * (2.**(-1./5) * ref_mass / mchirp)**(-5./6)


    def apply_prior_mods(self, priors):
        if (self.data_cfg.modification != None and isinstance(self.data_cfg.modification, list)) or self.aux:
            ## Check for known modifications
            # 1. bounded uniform signal duration (bounded_utau)
            # 2. bounded uniform chirp mass (bounded_umc)
            # 3. unbounded uniform signal duration (unbounded_utau)
            # 4. unbounded uniform chirp mass (unbounded_umc)
            # 5. bounded power law chirp mass (bounded_plmc)
            # 6. bounded power law signal duration (bounded_pltau)

            # Auxilliary prior mods
            if self.aux:
                ml = self.data_cfg.prior_low_mass
                mu = self.data_cfg.prior_high_mass
                mchirp_lower = (ml*ml / (ml+ml)**2.)**(3./5) * (ml + ml)
                mchirp_upper = (mu*mu / (mu+mu)**2.)**(3./5) * (mu + mu)
                if self.cflag.value in [0, 1]:
                    mchirp_upper = 25.0
                elif self.cflag.value in [2, 3]:
                    mchirp_lower = 25.0
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_uniform_mchirp_given_limits(mchirp_lower, mchirp_upper)

                # Distance and chirp distance
                chirp_distance_distr = UniformRadius(distance=(self.data_cfg.prior_low_chirp_dist, 
                                                            self.data_cfg.prior_high_chirp_dist))
                dchirp = np.asarray([chirp_distance_distr.rvs()[0][0] for _ in range(1)])
            
                # Get distance from chirp distance and chirp mass
                distance = self._dist_from_dchirp(dchirp, mchirp)

                ## Update Priors
                priors['q'] = q
                priors['mchirp'] = mchirp
                priors['mass1'] = mass1
                priors['mass2'] = mass2
                priors['chirp_distance'] = dchirp
                priors['distance'] = distance

                return priors

            # Pick the modification based on probability provided
            # Sum of probabilities must sum to 1
            current_probabilities = [self.modprobs[foo][self.epoch.value] for foo in self.data_cfg.modification]
            mod_thresholds = np.cumsum(current_probabilities)
            current_mod_idx = np.digitize(np.random.rand(1), mod_thresholds)[0]
            current_modification = self.data_cfg.modification[current_mod_idx]

            if current_modification == None:
                return priors

            if current_modification in ['bounded_utau']:
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_uniform_tau()
            
            elif current_modification in ['bounded_itau']:
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_importance_tau()

            elif current_modification in ['bounded_umc']:
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_uniform_mchirp()
            
            elif current_modification in ['bounded_pltau']:
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_powerlaw_tau()
                
            elif current_modification in ['bounded_plmc']:
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_powerlaw_mchirp()
            
            elif current_modification in ['template_placement_metric']:
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_template_placement_metric()
            
            elif current_modification in ['bounded_umcq']:
                mass1, mass2, q, mchirp, _ = self.bprior.get_bounded_gwparams_from_uniform_in_mchirp_q()
            
            elif current_modification in ['unbounded_utau', 'unbounded_umc']:
                _mass1, _mass2 = get_uniform_masses_with_mass1_gt_mass2(self.data_cfg.prior_low_mass, self.data_cfg.prior_high_mass, 1)
                # Masses used for mass ratio will not be used later as mass1 and mass2
                # We calculate them again after getting chirp mass
                q = q_from_mass1_mass2(_mass1, _mass2)
                # uniform signal duration
                if self.data_cfg.modification in ['unbounded_utau']:
                    tau_lower, tau_upper = get_tau_priors(self.data_cfg.prior_low_mass, 
                                                        self.data_cfg.prior_high_mass, 
                                                        self.data_cfg.signal_low_freq_cutoff)
                    # Get chirp mass
                    tau = np.random.uniform(tau_lower, tau_upper, 1)
                    mchirp = chirp_mass_from_signal_duration(tau, self.data_cfg.signal_low_freq_cutoff)
                # uniform chirp mass
                elif self.data_cfg.modification in ['unbounded_umc']:
                    mchirp_lower, mchirp_upper = get_mchirp_priors(self.data_cfg.prior_low_mass, 
                                                                self.data_cfg.prior_high_mass)
                    # Get chirp mass
                    mchirp = np.random.uniform(mchirp_lower, mchirp_upper, 1)
                
                # Common
                mass1, mass2 = mass1_mass2_from_mchirp_q(mchirp, q)
            
            else:
                if not self.aux:
                    raise ValueError("get_priors: Unknown modification specified in data_cfg.modification!")
            
            # Distance and chirp distance
            chirp_distance_distr = UniformRadius(distance=(self.data_cfg.prior_low_chirp_dist, 
                                                           self.data_cfg.prior_high_chirp_dist))
            dchirp = np.asarray([chirp_distance_distr.rvs()[0][0] for _ in range(1)])
        
            # Get distance from chirp distance and chirp mass
            distance = self._dist_from_dchirp(dchirp, mchirp)

            ## Update Priors
            priors['q'] = q
            priors['mchirp'] = mchirp
            priors['mass1'] = mass1
            priors['mass2'] = mass2
            priors['chirp_distance'] = dchirp
            priors['distance'] = distance
        
        elif self.data_cfg.modification != None and not isinstance(self.data_cfg.modification, str):
            raise ValueError("get_priors: Unknown data type used in data_cfg.modification!")
        
        return priors


    def get_waveform_parameters(self, seed):
        # Draw sample using config
        # Draw samples from prior distribution.
        np.random.seed(seed)
        # Draw samples from prior distribution.
        prior = self.randomsampler.rvs(size=1)
        # Apply parameter transformation.
        prior = transforms.apply_transforms(prior, self.waveform_transforms)
        # Apply modifications to prior (uniform signal duration/ uniform chirp mass)
        if self.training and np.random.random() <= self.data_cfg.modification_toggle_probability:
            prior = self.apply_prior_mods(prior)
        if not self.training and self.aux:
            prior = self.apply_prior_mods(prior)
        return prior


    def set_waveform_parameters(self, params, seed):
        # Get waveform params using PyCBC ini file
        priors = self.get_waveform_parameters(seed=seed)
        waveform_kwargs = {}
        waveform_kwargs['approximant'] = self.data_cfg.signal_approximant
        waveform_kwargs['f_ref'] = 20.0
        waveform_kwargs.update(dict(zip(list(priors.fieldnames), priors[0])))
        # Adding only dominant modes to the waveform
        # waveform_kwargs['mode_array'] = [ [2,2], [2,-2] ]
        # m1_msun, m2_msun, s1x, s1y, s1z, s2x, s2y, s2z, distance_mpc, tc, phiRef, inclination
        for key in params.keys():
            params[key] = priors[key][0]
        
        return (waveform_kwargs, params)


    def generate_data(self, target, seed, partial_signal=False):
        ## Generate waveform or read noise sample for D4
        # Target can be set using probablity of hypothesis
        targets = {}
        targets['gw'] = target
        
        if not target:
            ## Generate noise sample
            sample = self.noise_generation(self.special)
            # Dummy noise params
            params = self.params.copy()
            params['mchirp'] = -1
            # Dummy targets
            targets['norm_mchirp'] = -1
            targets['norm_tc'] = -1
        else:
            # Set parameters for waveform generation
            waveform_kwargs, params = self.set_waveform_parameters(self.params.copy(), seed)
            # Make a partial signal for non-astro samples
            if partial_signal:
                # NOTE (to future me): don't worry; targets['norm_tc'] will be removed in __getitem__
                # This sample will be labelled as noise and will have targets that reflect this 
                params['tc'] = np.random.uniform(low=0.1, high=11.9)
            ## Generate waveform sample
            sample = self.waveform_generation(waveform_kwargs, self.special)
            # Target params
            m1, m2 = params['mass1'], params['mass2']
            mchirp = (m1*m2 / (m1+m2)**2.)**(3./5) * (m1 + m2)
            params['mchirp'] = mchirp
            targets['norm_mchirp'] = self.norm['mchirp'].norm(mchirp)
            targets['norm_tc'] = self.norm['tc'].norm(params['tc'])
        
        # Generic params
        params['sample_rate'] = self.sample_rate
        params['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff

        return (sample, targets, params)
    

    def read_data(self, data_path):
        
        # Store all params within chunk file
        params = {}
        targets = {}
        
        # Get data from ExternalLink'ed lookup file
        HDF5_Dataset, didx = os.path.split(data_path)
        # Dataset Index should be an integer
        didx = int(didx)
        # Check whether data is signal or noise with target
        target = 1 if bool(re.search('signal', HDF5_Dataset)) else 0
        targets['gw'] = target
        # Access group
        group = self.extmain[HDF5_Dataset]
        
        if not target:
            ## Read noise data
            noise_1 = np.array(group['noise_1'][didx])
            noise_2 = np.array(group['noise_2'][didx])
            sample = np.stack([noise_1, noise_2], axis=0)
            # Dummy noise params
            targets['norm_mchirp'] = -1
            targets['norm_tc'] = -1
            # Dummy params
            params['mass1'] = -1
            params['mass2'] = -1
            params['distance'] = -1
            params['mchirp'] = -1
            params['dchirp'] = -1
            params['tc'] = -1
            params['network_snr'] = -1
        else:
            ## Read signal data
            h_plus = np.array(group['h_plus'][didx])
            h_cross = np.array(group['h_cross'][didx])
            sample = np.stack([h_plus, h_cross], axis=0)
            # Signal params
            params['start_time'] = group['start_time'][didx]
            params['interval_lower'] = group['interval_lower'][didx]
            params['interval_upper'] = group['interval_upper'][didx]
            params['mass1'] = group['mass1'][didx]
            params['mass2'] = group['mass2'][didx]
            params['distance'] = group['distance'][didx]
            params['mchirp'] = group['mchirp'][didx]
            params['tc'] = group['tc'][didx]
            params['network_snr'] = 0.0
            # Target params
            targets['norm_mchirp'] = group['norm_mchirp'][didx]
            targets['norm_tc'] = group['norm_tc'][didx]
        
        # Generic params
        params['sample_rate'] = self.sample_rate
        params['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
        
        return (sample, targets, params)


    def _augmentation_(self, sample, target, params, mode=None):
        """ Signal and Noise only Augmentation """
        if target and self.signal_only_transforms and mode=='signal':
            # Debug dir set to empty str
            sample, params, _special = self.signal_only_transforms(sample, params, self.special, '')
            self.special.update(_special)
        
        elif not target and self.noise_only_transforms and mode=='noise':
            sample = self.noise_only_transforms(sample, '')
        
        return sample, params
    
    
    def _noise_realisation_(self, sample, targets, seed=None):
        """ Finding random noise realisation for signal """
        if targets['gw']:
            # Read the noise data
            if self.data_cfg.OTF:
                pure_noise, targets_noise, params_noise = self.generate_data(target=0, seed=seed)
            else:
                random_noise_idx = random.choice(self.noise_idx)
                random_noise_data_path = self.data_paths[random_noise_idx]

                if self.data_cfg.check_generation:
                    if self.cfg.generation['noise'] != None:
                        pure_noise, targets_noise, params_noise = self.generate_data(target=0, seed=seed)
                    else:
                        pure_noise, targets_noise, params_noise = self.read_data(random_noise_data_path)
                else:
                    # Get data paths for external link
                    pure_noise, targets_noise, params_noise = self.read_data(random_noise_data_path)
            
            target_noise = targets_noise['gw']
            if self.training:
                pure_noise, _ = self._augmentation_(pure_noise, target_noise, params_noise, mode='noise')
            
            """ Adding noise to signals """
            if isinstance(pure_noise, np.ndarray) and isinstance(sample, np.ndarray):
                noisy_signal = sample + pure_noise
            else:
                raise TypeError('pure_signal or pure_noise is not an np.ndarray!')
            
        else:
            # If the sample is pure noise
            noisy_signal = sample
            pure_noise = sample
        
        return (noisy_signal, pure_noise)
    
    
    def _transforms_(self, sample, key=None):
        """ Transforms """
        # Apply transforms to signal and target (if any)
        if self.transforms:
            self.special['aux'] = self.cflag.value
            sample_transforms = self.transforms(sample, self.special, key=key)
        else:
            sample_transforms = {'sample': sample}
        
        return sample_transforms
    

    def target_handling(self, targets, params):
        # Gather up all parameters required for training and validation
        # NOTE: We can't use structured arrays here as PyTorch does not support it yet.
        #       Dictionaries are slow but it does the job.
        
        """ Targets """
        ## Storing targets as dictionary
        all_targets = {}
        # targets contain all other norm values for PE
        all_targets.update(targets)
        
        """ Source Parameters """
        ## Add sample params to all_targets variable
        source_params = {}
        # Distance and dchirp could have been alterred when rescaling SNR
        source_params.update(params)
        
        if targets['gw']:
            # Calculating the duration of the given signal
            lf = self.data_cfg.signal_low_freq_cutoff
            G = 6.67e-11
            c = 3.0e8
            source_params['signal_duration'] = 5. * (8.*np.pi*lf)**(-8./3.) * (params['mchirp']*1.989e30*G/c**3.)**(-5./3.)
            # Set chirp distance for signal
            source_params['dchirp'] = self._dchirp_from_dist(params['distance'], params['mchirp'])
        else:
            source_params['signal_duration'] = -1
            source_params['dchirp'] = -1
            source_params['network_snr'] = -1
        
        return all_targets, source_params
    

    def preprocess(self, sample, targets, params, seed):
        ## Signal Augmentation
        # Runs signal augmentation if sample is clean waveform
        pure_sample, params = self._augmentation_(sample, targets['gw'], params, mode='signal')

        ## Add noise realisation to the signals
        noisy_sample, pure_noise = self._noise_realisation_(pure_sample, targets, seed)
        
        ## Noise Augmentation
        if self.training:
            # Runs noise augmentation only for pure noise samples
            # TODO: var name noisy_sample might suggest that this is waveform + noise (fix this) 
            noisy_sample, params = self._augmentation_(noisy_sample, targets['gw'], params, mode='noise')
        
        return (noisy_sample, params)


    def __getitem__(self, idx):
        
        # Setting the unique seed for given sample
        if self.training:
            unique_epoch_seed = int((self.epoch.value*self.total_samples_per_epoch) + idx+1 + self.cfg.seed_offset_train)
            fixed_epoch_seed = int((self.total_samples_per_epoch) + idx+1)
        elif not self.training:
            unique_epoch_seed = int((self.epoch.value*self.total_samples_per_epoch) + idx+1  + self.cfg.seed_offset_valid) # 28 for all 1 epoch runs
            fixed_epoch_seed = int((self.total_samples_per_epoch) + idx+1 + 2**30)

        # Setting epoch number
        self.special['epoch'] = self.epoch.value
        
        # Set target for current iteration
        target = 1 if np.random.rand() < self.data_cfg.signal_probability else 0

        # Setting sample seed
        if target and self.data_cfg.fix_signal_seeds:
            seed = fixed_epoch_seed
        elif target and not self.data_cfg.fix_signal_seeds:
            seed = unique_epoch_seed
        elif not target and self.data_cfg.fix_noise_seeds:
            seed = fixed_epoch_seed
        elif not target and not self.data_cfg.fix_noise_seeds:
            seed = unique_epoch_seed
        else:
            raise ValueError('Seed options incorrect!')

        # Setting the seed for sample/iter
        # TODO: debug the fixed seed mode
        seed = unique_epoch_seed
        self.special['sample_seed'] = seed
        # Generate sample / read sample
        if self.data_cfg.OTF:
            # Timeslide mode is given tsmode_probability of all samples
            # So this mode should still maintain balance between normal signal and noise class samples
            if self.data_cfg.timeslide_mode and target == 0:
                make_nonastro_sample = 1 if np.random.rand() < self.data_cfg.tsmode_probability else 0
                # Two modes: mode_1=(signal + signal') or mode_2=(signal + noise)
                # Note: signal' need not be time coincident with signal.
                # So tc and tc' can be different
                non_astro_mode_select = 1 if np.random.rand() < self.data_cfg.non_astro_mode_select_probability else 2
            else:
                make_nonastro_sample = 0

            if not make_nonastro_sample:
                # Generate sample
                sample, targets, params = self.generate_data(target, seed=seed)
                # Signal/Noise class augmentation && add noise realisation (for signal class)
                noisy_sample, params = self.preprocess(sample, targets, params, seed)
            else:
                # All non-astro samples {01, 10, 12} are labelled at noise in targets['gw']
                # This will take unwanted network capacity and might reduce detection efficiency
                # TODO: Search for alternate solutions
                if non_astro_mode_select == 1:
                    ## Make signal + signal'
                    # We store both H1 and L1 for each sample for now
                    # After augmentation, we can remove the dummy samples
                    # Data now: [(H1, dummy_L1), (dummy_H1, L1)] --> [(H1, L1)]
                    sample, targets, params = self.generate_data(1, seed=seed, partial_signal=True)
                    noisy_sample_1, _ = self.preprocess(sample, targets, params, seed)
                    # Make signal' (tc' should automatically be different from tc)
                    seed_dash = seed+np.random.randint(1, 2**30)
                    sample, targets, params = self.generate_data(1, seed=seed_dash, partial_signal=True)
                    noisy_sample_2, _ = self.preprocess(sample, targets, params, seed_dash)
                    # Get required det samples from signal and signal'
                    noisy_sample = np.array([noisy_sample_1[0], noisy_sample_2[1]])

                elif non_astro_mode_select == 2:
                    # Make signal + noise
                    # I'm not okay with this. Labelling this as noise is not correct.
                    # Choice for H1 and L1: (noise + signal) or (signal + noise)
                    noise_det_choice = 0 if np.random.rand() < 0.5 else 1
                    # Make signal sample
                    sample, targets, params = self.generate_data(1, seed=seed, partial_signal=True)
                    signal_sample, _ = self.preprocess(sample, targets, params, seed)
                    # Make noise sample (this could be from H1 or L1)
                    seed_dash = seed+np.random.randint(1, 2**30)
                    sample, targets, params = self.generate_data(0, seed=seed_dash)
                    noise_sample, _ = self.preprocess(sample, targets, params, seed_dash)
                    # Get required det samples from signal and noise
                    if noise_det_choice == 0:
                        noisy_sample = np.array([noise_sample[0], signal_sample[1]])
                    else: 
                        noisy_sample = np.array([signal_sample[0], noise_sample[1]])
                    
                # Set params to noise params for non-astro samples
                params = self.params.copy()
                params['mchirp'] = -1
                params['sample_rate'] = self.sample_rate
                params['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
                # Set targets to noise
                targets = {}
                targets['gw'] = 0
                targets['norm_mchirp'] = -1
                targets['norm_tc'] = -1

        else:
            # Check if generation is present for signals or noise
            data_path = self.data_paths[idx]
            HDF5_Dataset, _ = os.path.split(data_path)
            # Check whether data is signal or noise with target
            current_target = 1 if bool(re.search('signal', HDF5_Dataset)) else 0
            if self.data_cfg.check_generation:
                cond_1 = self.cfg.generation['noise'] != None and not current_target
                cond_2 = self.cfg.generation['signal'] != None and current_target
                if cond_1 or cond_2:
                    sample, targets, params = self.generate_data(current_target, seed=seed)
                else:
                    sample, targets, params = self.read_data(data_path)
            else:
                # Get data paths for external link
                sample, targets, params = self.read_data(data_path)
        
            # Signal/Noise class augmentation && add noise realisation (for signal class)
            noisy_sample, params = self.preprocess(sample, targets, params, seed)

        ## Transformation Stage 1 (HighPass and Whitening)
        # These transformations are possible for pure noise before augmentation
        # NOTE: Cyclic shifting and phase augmentation cannot be performed before whitening
        #       The discontinuities in the sample will cause issues.
        # sample_transforms['sample'] will always provide the output of the last performed transform
        sample_transforms = self._transforms_(noisy_sample, key='stage1')

        ## Transformation Stage 2 (Multirate Sampling)
        # We keep this separate because many transformations require const delta t
        # MRsampling should be done as the last step of the transformation process
        mrsampling = self._transforms_(sample_transforms['sample'], key='stage2')

        # With the update 'sample' should point to mrsampled data
        # but still contain all other transformations in the correct key.
        sample_transforms.update(mrsampling)

        ## Targets and Parameters Handling
        all_targets, source_params = self.target_handling(targets, params)
        
        # Reducing memory footprint
        # This can only be performed after transforms and augmentation
        sample = np.array(sample_transforms['sample'], dtype=np.float32)
        
        # Tensorification
        # Convert signal/target to Tensor objects
        sample = torch.from_numpy(sample)

        # TODO: Clean this up!
        if not self.data_cfg.OTF:
            rem = ['start_time', 'interval_lower', 'interval_upper', 'declination', 'right_ascension', 'polarisation_angle']
            rem += ['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', 'coa_phase', 'inclination']
            for rm_param in rem:
                if rm_param in source_params.keys():
                    source_params.pop(rm_param)
        
        return (sample, all_targets, source_params)
