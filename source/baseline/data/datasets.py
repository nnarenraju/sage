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
import shutil
import random
import numpy as np
from torch.utils.data import Dataset

# LOCAL
from data.snr_calculation import get_network_snr
from data.plot_dataloader_unit import plot_unit
from data.multirate_sampling import get_sampling_rate_bins

# PyCBC
import pycbc
from pycbc import distributions
from pycbc.types import FrequencySeries

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

class MLMDC1(Dataset):
    
    def __init__(self, data_paths, targets, transforms=None, target_transforms=None,
                 signal_only_transforms=None, noise_only_transforms=None, 
                 training=False, testing=False, store_device='cpu', train_device='cpu', 
                 cfg=None, data_cfg=None):
        
        super().__init__()
        self.data_paths = data_paths
        self.targets = targets
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.signal_only_transforms = signal_only_transforms
        self.noise_only_transforms = noise_only_transforms
        self.store_device = store_device
        self.train_device = train_device
        self.cfg = cfg
        self.data_cfg = data_cfg
        self.data_loc = os.path.join(self.data_cfg.parent_dir, self.data_cfg.data_dir)
        
        self.training = training
        
        # Set CUDA device for pin_memory if needed
        if bool(re.search('cuda', self.cfg.store_device)):
            setattr(self, 'foo', torch.cuda.set_device(self.cfg.store_device))
        
        
        """ PSD Handling (used in whitening) """
        # Store the PSD files here in RAM. This reduces the overhead when whitening.
        # Read all psds in the data_dir and store then as FrequencySeries
        self.PSDs = {}
        psd_files = glob.glob(os.path.join(self.data_loc, "psds/*"))
        for psd_file in psd_files:
            with h5py.File(psd_file, 'r') as fp:
                data = np.array(fp['data'])
                delta_f = fp.attrs['delta_f']
                name = fp.attrs['name']
                
            psd_data = FrequencySeries(data, delta_f=delta_f)
            # Store PSD data into lookup dict
            self.PSDs[name] = psd_data
        
        if self.data_cfg.dataset == 1:
            self.psds_data = [self.PSDs['aLIGOZeroDetHighPower']]*2
        else:
            self.psds_data = [self.PSDs['median_det1'], self.PSDs['median_det2']]
            
        
        """ Multi-rate Sampling """
        # Get the sampling rates and their bins idx
        data_cfg.dbins = get_sampling_rate_bins(data_cfg)
        
        
        """ LAL Detector Objects (used in project_wave - AugPolSky) """
        # Detector objects (these are lal objects and may present problems when parallelising)
        # Create the detectors (TODO: generalise this!!!)
        detectors_abbr = ('H1', 'L1')
        self.detectors = []
        for det_abbr in detectors_abbr:
            self.detectors.append(pycbc.detector.Detector(det_abbr))
        
        
        """ PyCBC Distributions (used in AugPolSky and AugDistance) """
        ## Distribution objects for augmentation
        # Used for obtaining random polarisation angle
        self.uniform_angle_distr = distributions.angular.UniformAngle(uniform_angle=(0., 2.0*np.pi))
        # Used for obtaining random ra and dec
        self.skylocation_distr = distributions.sky_location.UniformSky()
        # Used for obtaining random mass
        self.mass_distr = distributions.Uniform(mass=(data_cfg.prior_low_mass, data_cfg.prior_high_mass))
        # Used for obtaining random chirp distance
        dist_gen = distributions.power_law.UniformRadius
        self.chirp_distance_distr = dist_gen(distance=(data_cfg.prior_low_chirp_dist, 
                                                       data_cfg.prior_high_chirp_dist))
        # Distributions object
        self.distrs = {'pol': self.uniform_angle_distr, 'sky': self.skylocation_distr,
                       'mass': self.mass_distr, 'dchirp': self.chirp_distance_distr}
        
        
        """ Normalising the augmented params (if needed) """
        # Normalise chirp mass
        ml = self.data_cfg.prior_low_mass
        mu = self.data_cfg.prior_high_mass
        # m2 will always be slightly lower than m1, but (m, m) will give limit
        # that the mchirp will never reach but tends to as num_samples tends to inf.
        # Range for mchirp can be written as --> (min_mchirp, max_mchirp)
        min_mchirp = (ml*ml / (ml+ml)**2.)**(3./5) * (ml + ml)
        max_mchirp = (mu*mu / (mu+mu)**2.)**(3./5) * (mu + mu)
        
        # Get distance ranges from chirp distance priors
        # mchirp present in numerator of self.distance_from_chirp_distance.
        # Thus, min_mchirp for dist_lower and max_mchirp for dist_upper
        dist_lower = self._dist_from_dchirp(self.data_cfg.prior_low_chirp_dist, min_mchirp)
        dist_upper = self._dist_from_dchirp(self.data_cfg.prior_high_chirp_dist, max_mchirp)
        # get normlised distance class
        self.norm_dist = Normalise(min_val=dist_lower, max_val=dist_upper)
        
        # Normalise chirp distance
        self.norm_dchirp = Normalise(min_val=self.data_cfg.prior_low_chirp_dist, 
                                     max_val=self.data_cfg.prior_high_chirp_dist)
        # Normalise the mass ratio 'q'
        self.norm_q = Normalise(min_val=1.0, max_val=mu/ml)
        self.norm_invq = Normalise(min_val=0.0, max_val=1.0)
        
        # Normalise the SNR
        self.norm_snr = Normalise(min_val=self.cfg.rescaled_snr_lower,
                                  max_val=self.cfg.rescaled_snr_upper)
        
        # All normalisation variables
        self.norm = {}
        self.norm['dist'] = self.norm_dist
        self.norm['dchirp'] = self.norm_dchirp
        self.norm['q'] = self.norm_q
        self.norm['invq'] = self.norm_invq
        self.norm['snr'] = self.norm_snr
        
        
        """ Data Save Params (for plotting sample just before training) """
        if self.data_cfg.num_sample_save == None:
            self.num_sample_save = 100
        else:
            self.num_sample_save = data_cfg.num_sample_save
        
        
        """ Random noise realisation """
        self.noise_idx = np.argwhere(self.targets == 0).flatten()
        self.noise_norm_idx = np.arange(len(self.noise_idx))
        self.noise_paths = self.data_paths[self.noise_idx]
        
        
        """ Keep ExternalLink Lookup table open till end of run """
        lookup = os.path.join(cfg.export_dir, 'extlinks.hdf')
        self.extmain = h5py.File(lookup, 'r', libver='latest')
        self.sample_rate = self.extmain.attrs['sample_rate']
        self.noise_low_freq_cutoff = self.extmain.attrs['noise_low_freq_cutoff']
        
        """ numpy random """
        self.np_gen = np.random.default_rng()
        
        self.debug = cfg.debug
        if self.debug:
            self.debug_dir = os.path.join(cfg.export_dir, 'DEBUG')
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir, exist_ok=False)
        else:
            self.debug_dir = ''
        

    def __len__(self):
        return len(self.data_paths)

    
    def _read_(self, data_path):
        
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
            targets['norm_dchirp'] = -1
            targets['norm_dist'] = -1
            targets['norm_mchirp'] = -1
            targets['norm_q'] = -1
            targets['norm_invq'] = -1
            targets['norm_tc'] = -1
            # Dummy params
            params['mass1'] = -1
            params['mass2'] = -1
            params['distance'] = -1
            params['mchirp'] = -1
            params['dchirp'] = -1
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
            # Target params
            targets['norm_dchirp'] = group['norm_dchirp'][didx]
            targets['norm_dist'] = group['norm_dist'][didx]
            targets['norm_mchirp'] = group['norm_mchirp'][didx]
            targets['norm_q'] = group['norm_q'][didx]
            targets['norm_invq'] = group['norm_invq'][didx]
            targets['norm_tc'] = group['norm_tc'][didx]
        
        # Generic params
        params['sample_rate'] = self.sample_rate
        params['noise_low_freq_cutoff'] = self.noise_low_freq_cutoff
        
        return (sample, targets, params)
    
    
    def _dist_from_dchirp(self, chirp_distance, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the luminosity distance given a chirp distance and chirp mass.
        return chirp_distance * (2.**(-1./5) * ref_mass / mchirp)**(-5./6)
    
    
    def _dchirp_from_dist(self, dist, mchirp, ref_mass=1.4):
        # Credits: https://pycbc.org/pycbc/latest/html/_modules/pycbc/conversions.html
        # Returns the chirp distance given the luminosity distance and chirp mass.
        return dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)
    
    
    def _augmentation_(self, sample, target, params, debug):
        """ Signal and Noise only Augmentation """
        ## Convert the signal from h_plus and h_cross to h_t and augment
        # Get augmented values if they affect the parameter estimation labels
        augmented_labels = {}
        if target and self.signal_only_transforms:
            aug_sample = self.signal_only_transforms(sample, self.detectors, self.distrs, self.debug_dir, **params)
            sample = aug_sample['signal']
            # Change the values of augmented parameters for sample and update corresponding labels (if needed)
            augkeys = [foo for foo in aug_sample.keys() if foo != 'signal']
            for augkey in augkeys:
                augmented_labels['norm_'+augkey] = self.norm[augkey].norm(aug_sample[augkey])
                # Update the distance parameter according to the new augmented distance
                # Generalise this to include all params
                if augkey == 'dist':
                    params['distance'] = aug_sample[augkey]
                
        elif not target and self.noise_only_transforms:
            sample = self.noise_only_transforms(sample, self.debug_dir)
        
        return sample, augmented_labels, params
    
    
    def _noise_realisation_(self, sample, target, params):
        """ Finding random noise realisation for signal """
        # Random noise realisation is the only procedure available
        # Fixed noise realisation was deprecated
        
        if self.cfg.add_random_noise_realisation and target:
            # Pick a random noise realisation to add to the signal
            random_noise_idx = random.choice(self.noise_idx)
            if self.debug:
                debug_idx = np.argwhere(self.noise_idx == random_noise_idx).flatten()
                if self.training:
                    filename = 'save_augment_train_random_noise_idx.txt'
                else:
                    filename = 'save_augment_valid_random_noise_idx.txt'
                with open(os.path.join(self.debug_dir, filename), 'a') as fp:
                    string = "{} ".format(self.noise_norm_idx[debug_idx][0])
                    fp.write(string)
                    
            random_noise_data_path = self.data_paths[random_noise_idx]
            
            # Read the noise data
            pure_noise, _, _ = self._read_(random_noise_data_path)
            
            """ Calculation of Network SNR (use pure signal, before adding noise realisation) """
            prelim_network_snr = get_network_snr(sample, self.psds_data, params, self.cfg.export_dir, self.debug)
            
            """ Adding noise to signals """
            if isinstance(pure_noise, np.ndarray) and isinstance(sample, np.ndarray):
                if self.cfg.rescale_snr:
                    # Rescaling the SNR to a uniform distribution within a given range
                    target_snr = self.np_gen.uniform(self.cfg.rescaled_snr_lower, self.cfg.rescaled_snr_upper)
                    rescaling_factor = target_snr/prelim_network_snr
                    # Add noise to rescaled signal
                    noisy_signal = pure_noise + (sample * rescaling_factor)
                    # Adjust distance parameter for signal according to the new rescaled SNR
                    rescaled_distance = params['distance'] / rescaling_factor
                    rescaled_dchirp = self._dchirp_from_dist(rescaled_distance, params['mchirp'])
                    # Update targets and params with new rescaled distance is not possible
                    # We do not know the priors of network_snr properly
                    if 'norm_dist' in self.cfg.parameter_estimation or 'norm_dchirp' in self.cfg.parameter_estimation:
                        raise RuntimeError('rescale_snr option cannot be used with dist/dchirp PE!')
                    # Update final network SNR to new value given by target SNR
                    network_snr = target_snr
                    norm_snr = self.norm_snr.norm(network_snr)
                    # Update the params dictionary with new rescaled distances
                    params['distance'] = rescaled_distance
                    params['dchirp'] = rescaled_dchirp
                else:
                    network_snr = prelim_network_snr
                    if 'norm_snr' in self.cfg.parameter_estimation:
                        raise RuntimeError('rescale_snr option is off. Cannot use norm_snr PE!')
                    norm_snr = -1
                    noisy_signal = sample + pure_noise
                    
            else:
                raise TypeError('pure_signal or pure_noise is not an np.ndarray!')
            
        elif not self.cfg.add_random_noise_realisation and target:
            # Fixed noise realisation to add to the signal
            raise DeprecationWarning('Fixed noise realisation feature deprecated on June 8th, 2022')
            
        else:
            # If the sample is pure noise
            noisy_signal = sample
            pure_noise = None
            # For pure noise sample, we could calculate the matched filter SNR and set a target
            # This value will be very low, but could help improve FAR
            if self.cfg.network_snr_for_noise:
                raise NotImplementedError('SNR for noise samples under construction!')
                network_snr = get_network_snr(sample, self.psds_data, params, None, False)
                norm_snr = self.norm_snr.norm(network_snr)
            else:
                network_snr = -1
                norm_snr = -1
        
        return (noisy_signal, pure_noise, network_snr, norm_snr, params)
    
    
    def _transforms_(self, noisy_sample, target):
        """ Transforms """
        # Apply transforms to signal and target (if any)
        if self.transforms:
            sample = self.transforms(noisy_sample, self.psds_data, self.data_cfg)
        else:
            sample = noisy_sample
            
        if self.target_transforms:
            target = self.target_transforms(target)
        
        return (sample, target)
    
    
    def _plotting_(self, pure_sample, pure_noise, noisy_sample, trans_noisy_sample, network_snr, idx, params):
        """ Plotting idx data (if flag is set to True) """
        # Input parameters
        if self.transforms:
            trans_pure_signal = self.transforms(pure_sample, self.psds_data, self.data_cfg)
        else:
            trans_pure_signal = None
        
        save_path = self.cfg.export_dir
        data_dir = os.path.normpath(self.data_loc).split(os.path.sep)[-1]
        # Plotting unit data
        plot_unit(pure_sample, pure_noise, noisy_sample, trans_pure_signal, trans_noisy_sample, 
                  params['mass1'], params['mass2'], network_snr, params['sample_rate'],
                  save_path, data_dir, idx)
    
    
    def __getitem__(self, idx):
        
        data_path = self.data_paths[idx]
        
        ## Read the sample(s)
        sample, targets, params = self._read_(data_path)
        target_gw = targets['gw']
        
        ## Signal and Noise Augmentation
        pure_sample, aug_labels, params = self._augmentation_(sample, target_gw, params, self.debug)
        ## Add noise realisation to the signals
        noisy_sample, pure_noise, network_snr, norm_snr, params = self._noise_realisation_(pure_sample, target_gw, params)
        
        ## Target handling
        target_gw = np.array([target_gw])
        
        ## Transforms
        sample, target_gw = self._transforms_(noisy_sample, target_gw)
        
        # Storing target as dictionaries
        all_targets = {}
        all_targets['norm_snr'] = norm_snr
        all_targets.update(targets)
        
        # Update parameter labels if augmentation changed them
        # aug_labels must have the same keys as targets dict
        all_targets.update(aug_labels)
        
        # Add sample params to all_targets variable
        source_params = {}
        source_params['snr'] = network_snr
        # Distance and dchirp could have been alterred when rescaling SNR
        source_params['distance'] = params['distance']
        if 'dchirp' in params.keys():
            source_params['dchirp'] = params['dchirp']
        else:
            source_params['dchirp'] = self._dchirp_from_dist(params['distance'], params['mchirp'])
        # Other params should be unalterred
        source_params['mchirp'] = params['mchirp']
        source_params['mass1'] = params['mass1']
        source_params['mass2'] = params['mass2']
        if target_gw[0]:
            # Written as m2/m1. Different from PyCBC format of m1/m2. m1>m2 in both cases.
            source_params['q'] = params['mass2']/params['mass1']
            # Calculating the duration of the given signal
            lf = self.data_cfg.signal_low_freq_cutoff
            source_params['signal_duration'] = 5. * (8.*np.pi*lf)**(-8./3.) * params['mchirp']**(-5./3.)
        else:
            source_params['q'] = -1
            source_params['signal_duration'] = -1
        
        ## Plotting
        if self.debug:
            check_dir = os.path.join(self.cfg.export_dir, 'SAMPLES')
            if os.path.isdir(check_dir):
                check_path = os.path.join(check_dir, '*.png')
                num_created = len(glob.glob(check_path))
            else:
                num_created = 0
                
            if target_gw[0] and num_created < self.cfg.num_sample_save:
                self._plotting_(pure_sample, pure_noise, noisy_sample, sample, network_snr, idx, params)
        
        
        """ Reducing memory footprint """
        # This can only be performed after transforms and augmentation
        sample = np.array(sample, dtype=np.float32)
        
        """ Tensorification """
        # Convert signal/target to Tensor objects
        sample = torch.from_numpy(sample)
        
        return (sample, all_targets, source_params)



""" Testing Dataloaders """






""" Other Loaders """

class Simple(Dataset):
    """
    Simple read-and-load-type dataset object
    WARNING!!!: This custom dataset *cannot* be used in configs as primary dataset object
    
    """
    
    def __init__(self, samples, targets, store_device='cpu', train_device='cpu'):
        
        super().__init__()
        self.samples = samples
        self.targets = targets
        self.store_device = store_device
        self.train_device = train_device
        assert len(self.samples) == len(self.targets)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        """ Tensorification and Device Compatibility """
        # Convert signal/target to Tensor objects and set device and data_type
        global tensor_dtype
        sample = self.samples[idx].to(dtype=tensor_dtype, device=self.train_device)
        target = self.targets[idx].to(dtype=tensor_dtype, device=self.train_device)
        
        return (sample, target)
