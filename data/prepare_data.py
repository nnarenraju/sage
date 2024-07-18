# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Nov 27 14:39:57 2021

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
import h5py
import torch
import shutil
import configparser

import numpy as np
import pandas as pd
import torch.utils.data as D

# Shared epopch value
from multiprocessing import Value
from functools import partial

# Class balanced splitting
from sklearn.model_selection import train_test_split

# LOCAL
from configs import *
from data_configs import *
from data.multirate_sampling import get_sampling_rate_bins_type1, get_sampling_rate_bins_type2
from data.multirate_sampling import multirate_sampling


def _cfg_to_ini_():
    # Name conversion from data_cfg to pycbc ini
    cfg_to_ini = {}
    cfg_to_ini['reference_freq'] = [('f_ref', 'static_params')]
    cfg_to_ini['signal_low_freq_cutoff'] = [('f_lower', 'static_params')]
    cfg_to_ini['signal_approximant'] = [('approximant', 'static_params')]
    cfg_to_ini['tc_inject_lower'] = [('min-tc', 'prior-tc')]
    cfg_to_ini['tc_inject_upper'] = [('max-tc', 'prior-tc')]
    cfg_to_ini['prior_low_mass'] = [('min-mass1', 'prior-mass1'), ('min-mass2', 'prior-mass2')]
    cfg_to_ini['prior_high_mass'] = [('max-mass1', 'prior-mass1'), ('max-mass2', 'prior-mass2')]
    cfg_to_ini['prior_low_chirp_dist'] = [('min-chirp_distance', 'prior-chirp_distance')]
    cfg_to_ini['prior_high_chirp_dist'] = [('max-chirp_distance', 'prior-chirp_distance')]
    return cfg_to_ini


class DataModule:
    """
    Module to handle dataset preparation.
    Not to be confused with PyTorch Lightning DataModule (L-DM).
    # Message written on 27/11/2021
    Due to lack of compatibility with cross-validation, L-DM was not used
    
    User-Defined Functions
    ----------------------
    configure_path - 
    get_summary - 
    get_metadata - 
    get_dataset_objects - 
    get_data_loader -
    
    """

    def __init__(self):
        pass

    def configure_pipeline(opts):
        """ 
        Read pipeline configuration and create export directory
        
        Arguments
        ---------
        opts : object
            argparse object from config.py
        
        Returns
        -------
        cfg : object
            model object from config.py
            
        """
        
        # cfg contains the model class (see config.py)
        # This creates a new instance, handle this in run.py
        cfg = eval(opts.config)
        return cfg

    
    def configure_dataset(opts):
        """ 
        Read dataset configuration
        
        Arguments
        ---------
        opts : object
            argparse object from data_config.py
        
        Returns
        -------
        cfg : object
            model object from data_config.py
            
        """

        # data_cfg contains the dataset configuration
        data_cfg = eval(opts.data_config)
        # Generating pycbc ini file from template
        cfg_to_ini = _cfg_to_ini_()
        config = configparser.ConfigParser()
        ini_template_path = './ini_files/templates/ds{}_template.ini'.format(data_cfg.dataset)
        ini_file_path = './ini_files/ds{}.ini'.format(data_cfg.dataset)
        config.read(ini_template_path)
        for cfg_mdata, ini_mdata in cfg_to_ini.items():
            for varname, clsname in ini_mdata:
                value = str(getattr(data_cfg, cfg_mdata))
                config.set(clsname, varname, value)

        with open(ini_file_path, 'w') as configfile:
            config.write(configfile)

        return data_cfg
    
    
    def make_export_dir(cfg):
        # Results directory
        if not os.path.exists(cfg.export_dir):
            cfg.export_dir.mkdir(parents=True, exist_ok=False)
        else:
            raise IOError('Export directory already exists!')
    
    
    def get_summary(cfg, data_cfg, export_dir):
        """
        Creates/uses a training dataset in hdf format
        Consolidate the IDs, target and path into CSV files
        Done for training and testing data
    
        Parameters
        ----------
        data_cfg : class
            Class containing data_config information
            eg. cfg is Default class in data_configs.py
        
        export_dir : str
            Location of output directory for results and lightning
    
        Returns
        -------
        None.
    
        """
        
        if not data_cfg.OTF:
            # Getting the attributes of data_config class as dict
            dc_attrs = {key:value for key, value in data_cfg.__dict__.items() if not key.startswith('__') and not callable(key)}
            
            # Set "make_dataset" to the appropriate method
            make_module = dc_attrs['make_module']
            
            # Training data
            if dc_attrs['make_dataset']:
                make_module(dc_attrs, export_dir)
            
            # Check if dataset already exists
            # If it does, check if training.hdf exists within it and save in export dir
            check_dir = os.path.join(dc_attrs['parent_dir'], dc_attrs['data_dir'])
            if os.path.isdir(check_dir):
                # Lookup table formats
                elink_file = os.path.join(check_dir, "extlinks.hdf")
                # Copy lookup tables to export_dir
                if os.path.isfile(elink_file):
                    # Move the extlinks.hdf to export_dir for pipeline
                    move_location = os.path.join(export_dir, 'extlinks.hdf')
                    shutil.copy(elink_file, move_location)
            else:
                raise FileNotFoundError(f"prepare_data.get_summary: {check_dir} not found!")
        else:
            print('Running ORChiD in OTF Mode')
            
    
    def get_metadata(cfg, data_cfg):
        """
        Retrieve dataset metadata
        
        Parameters
        ----------
        cfg : object
            model object from config.py
        
        Returns
        -------
        train : pd dataframe
            training metadata from input dir
        test : pd dataframe
            testing metadata from input dir
        folds : K-fold split data idx
            train/test indices to split data in train/test sets    
            
        """
        
        # Set lookup table ('training.hdf' lookup table is also present for batch-loading)
        lookup_file = 'extlinks.hdf'
        
        lookup_table = os.path.join(cfg.export_dir, lookup_file)
        # Deprecated: Usage of self.get_metadata with make_default_dataset (non-MP)
        with h5py.File(lookup_table, 'a') as fp:
            ids = np.array(fp['id'][:])
            paths = np.array([foo.decode('utf-8') for foo in fp['path']])
            targets = np.array(fp['target'][:])
            dstype = np.array([foo.decode('utf-8') for foo in fp['type']])
            mchirp = np.array(fp['mchirp'][:])
        
        # Chunk the entire lookup table for BatchLoader method used within MLMDC1 dataset object
        # The independant BatchLoader Method has been deprecated as of May 24th, 2022
        # mega_batch_size method deprecated on May 31st, 2022
        
        """
        Deprecated MegaBatch method
        ---------------------------
            
            if cfg.mega_batch_size != -1 and cfg.mega_batch_size != None:
                paths = np.array_split(paths, int(len(paths)/cfg.mega_batch_size))
                targets = np.array_split(targets, int(len(targets)/cfg.mega_batch_size))
                # ids are not split, they are converted to regular single int ids
                ids = np.arange(len(targets))
                
        """
        
        # Create a consolidated lookup date as a Pandas Dataframe
        lookup = list(zip(ids, paths, targets, dstype, mchirp))
        train = pd.DataFrame(lookup, columns=['id', 'path', 'target', 'dstype', 'mchirp'])
        
        # if debug, use a data subset
        if cfg.debug:
            # Initial debug split has to be class balanced as well
            idxs = np.arange(len(train))
            all_targets = train['target'].values
            # Compure the subset proportion based
            if cfg.debug_size < len(train):
                prop = cfg.debug_size/len(train)
            else:
                raise ValueError('cfg.debug_size > full dataset size!')
            
            if data_cfg.dataset in [1, 2, 3]:
                _, X, _, y = train_test_split(idxs, all_targets, test_size=prop, 
                                              random_state=42, stratify=all_targets)
                train = train.iloc[X]
                
            elif data_cfg.dataset == 4: 
                num_training = int(0.8*cfg.debug_size)
                num_validation = int(0.2*cfg.debug_size)
                training_data = train.loc[train['dstype'] == 'training'].head(num_training)
                balance_params = {'mchirp': training_data['mchirp'].values}
                validation_data = train.loc[train['dstype'] == 'validation'].head(num_validation)
                # Get the required number of samples from each
                train = pd.concat([training_data, validation_data])
                folds = [(np.arange(len(training_data)), np.arange(len(training_data), len(training_data)+len(validation_data)))]

                return (train, folds, balance_params)
        
        ## Splitting
        if cfg.splitter is not None:
            # Function ensures equal ratio of all classes in each fold
            folds = list(cfg.splitter.split(train, train['target'].values))
        else:
            # Splitting training and validation in 80-20 sections
            # This essentially has all the data in 1 fold
            idxs = np.arange(len(train))
            # Use idxs as training data together with targets to stratify into train and test set
            # This ensures a class balanced training and testing dataset
            all_targets = train['target'].values
            
            if data_cfg.dataset in [1, 2, 3]:
                X_train, X_valid, _, _ = train_test_split(idxs, all_targets, test_size=0.2, 
                                                          random_state=42, stratify=all_targets)
                # Save as folds for training and validation 
                balance_params = None
                folds = [(X_train, X_valid)]
            
            elif data_cfg.dataset == 4:
                if cfg.subset_for_funsies: 
                    # We need to ignore balanced dataset for this to work (can't be bothered to fix this)
                    num_training = int(0.8*cfg.debug_size)
                    num_validation = int(0.2*cfg.debug_size)
                    training_data = train.loc[train['dstype'] == 'training'].head(num_training)
                    balance_params = {'mchirp': training_data['mchirp'].values}
                    validation_data = train.loc[train['dstype'] == 'validation'].head(num_validation)
                    # Get the required number of samples from each
                    train = pd.concat([training_data, validation_data])
                    folds = [(np.arange(len(training_data)), np.arange(len(training_data), len(training_data)+len(validation_data)))]
                else:
                    training_data = train.loc[train['dstype'] == 'training']
                    balance_params = {'mchirp': training_data['mchirp'].values}
                    validation_data = train.loc[train['dstype'] == 'validation']
                    # Get the required number of samples from each
                    folds = [(training_data['id'].values, validation_data['id'].values)]
                
        return (train, folds, balance_params)
    
    
    def get_dataset_objects(cfg, data_cfg, train_fold, valid_fold):
        """
        Initialise Training/Validation/Testing Dataset
        
        Options provided in dataset object:
            [1] data_paths, targets
            [2] transforms=None, target_transforms=None
            
            TODO: implement cache data option
            [3] cache=None
        
        Link to map-style dataset object:
            https://pytorch.org/docs/stable/data.html
        
        Parameters
        ----------
        cfg : object
            model object from config.py
        
        Returns
        -------
        train_dataset : map-style dataset object
        valid_dataset : map-style dataset object
        
        """
        # Create a Weighted Random Sampler for keeping class balance between batches
        # StackOverflow Link: https://stackoverflow.com/questions/62878940/how-to-create-a-balancing-cycling-iterator-in-pytourch
        # Where weights is the probability of each sample, it depends in how many samples
        # per category you have, for instance, if you data is simple as that data = [0, 1, 0, 0, 1],
        # class '0' count is 3, and class '1' count is 2 So weights vector is [1/3, 1/2, 1/3, 1/3, 1/2].
        if not data_cfg.OTF:
            ttargets = train_fold['target'].values
            check_class_balance = len(ttargets[ttargets == 1])/len(ttargets)
            if check_class_balance != 0.5 and not cfg.ignore_dset_imbalance:
                raise ValueError('Encountered a class imbalanced (num_signals/tot = {}) training dataset!'.format(check_class_balance))
            
            # Set data_paths and targets
            data_paths = train_fold['path'].values
            targets = train_fold['target'].values
        
        else:
            # Set data_paths and targets for OTF
            data_paths = targets = None
        
        # Create the training and validation dataset objects
        train_dataset = cfg.dataset(
                data_paths=data_paths, targets=targets,
                waveform_generation=cfg.generation['signal'], noise_generation=cfg.generation['noise'],
                transforms=cfg.transforms['train'], target_transforms=cfg.transforms['target'],
                signal_only_transforms=cfg.transforms['signal'], 
                noise_only_transforms=cfg.transforms['noise'],
                training = True, cfg=cfg, data_cfg=data_cfg, store_device=cfg.store_device,
                train_device=cfg.train_device, **cfg.dataset_params)
        
        # Validation dataset
        if not data_cfg.OTF:
            vtargets = valid_fold['target'].values
            check_class_balance = len(vtargets[vtargets == 1])/len(vtargets)
            if check_class_balance != 0.5 and not cfg.ignore_dset_imbalance:
                raise ValueError('Encountered a class imbalanced (num_signals/tot = {}) validation dataset!'.format(check_class_balance))
            
            # Set data_paths and targets
            data_paths = valid_fold['path'].values
            targets = valid_fold['target'].values
        
        else:
            # Set data_paths and targets for OTF
            data_paths = targets = None
        
        valid_dataset = cfg.dataset(
                data_paths=data_paths, targets=targets,
                waveform_generation=cfg.generation['signal'], noise_generation=cfg.generation['noise'],
                transforms=cfg.transforms['test'], target_transforms=cfg.transforms['target'],
                signal_only_transforms=cfg.transforms['signal'],
                noise_only_transforms=cfg.transforms['noise'],
                training=False, cfg=cfg, data_cfg=data_cfg, store_device=cfg.store_device,
                train_device=cfg.train_device, **cfg.dataset_params)
        
        # AUX validation sets
        aux_dataset = cfg.dataset(
                data_paths=data_paths, targets=targets,
                waveform_generation=cfg.generation['signal'], noise_generation=cfg.generation['noise'],
                transforms=cfg.transforms['test'], target_transforms=cfg.transforms['target'],
                signal_only_transforms=cfg.transforms['signal'],
                noise_only_transforms=cfg.transforms['noise'],
                training=False, aux=True, cfg=cfg, data_cfg=data_cfg, store_device=cfg.store_device,
                train_device=cfg.train_device, **cfg.dataset_params)
        
        return (train_dataset, valid_dataset, aux_dataset)
    

    def get_dataloader(cfg, train_data, valid_data, aux_data, balance_params):
        """
        Create Pytorch DataLoader objects
        
        Signature of DataLoader:
        DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
                   batch_sampler=None, num_workers=0, collate_fn=None,
                   pin_memory=False, drop_last=False, timeout=0,
                   worker_init_fn=None, *, prefetch_factor=2,
                   persistent_workers=False)
        
        Parameters
        ----------
        train_data : map-style dataset object
        valid_data : map-style dataset object
    
        Returns
        -------
        train_loader : Pytorch DataLoader object
        valid_loader : Pytorch DataLoader object
        
        """
        
        def init_fn(epoch_n, cflag, worker_id):
            info = D.get_worker_info()
            info.dataset.epoch = epoch_n
            info.dataset.cflag = cflag

        def make_weights_for_balancing_param(param, nbins=8):
            # Remove the -1 values from the array and count them separately
            # Sum of all weights for noise and signals should be the same
            counts, bin_edges = np.histogram(param, bins=nbins)
            counts = np.insert(counts, 1, 0)
            weight_per_bin = [len(param)/count for count in counts if count > 0.0]
            bin_edges = np.insert(bin_edges, 1, 0)
            weight_per_bin.insert(1, 0)
            idxs = np.digitize(param, bin_edges, right=True)
            idxs[idxs == 0] = 1
            idxs = idxs - 1
            weights = [weight_per_bin[idx] for idx in idxs]
            weights = torch.DoubleTensor(weights) 
            return weights

        epoch_n = Value('i', -1)
        cflag = Value('i', -1)

        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        pin_memory = cfg.pin_memory
        persistent_workers = cfg.persistent_workers
        
        # Sometimes in MAC systems, setting num_workers > 0 causes a intraop warning to appear
        # This does not seem to produce any incorrect results. However it is worrying.
        # Temporary fix (suppresses the error, but does not fix the underlying issue)
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # NOTE: Use of tsampler and vsampler has been deprecated as of: January 18th, 2023
        # NOTE: Use of sampler reinstated for getting rid of bias on mchirp: July 12th, 2023
        # weights = make_weights_for_balancing_param(balance_params['mchirp'], nbins=8)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = D.DataLoader(
            train_data, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, shuffle=True,
            prefetch_factor=cfg.prefetch_factor, persistent_workers=persistent_workers,
            worker_init_fn=partial(init_fn, epoch_n, cflag))
        
        valid_loader = D.DataLoader(
            valid_data, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
            prefetch_factor=cfg.prefetch_factor, persistent_workers=persistent_workers,
            worker_init_fn=partial(init_fn, epoch_n, cflag))
        
        aux_loader = D.DataLoader(
            aux_data, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
            prefetch_factor=cfg.prefetch_factor, persistent_workers=persistent_workers,
            worker_init_fn=partial(init_fn, epoch_n, cflag))
        
        return (train_loader, valid_loader, aux_loader, epoch_n, cflag)
 

    def input_sample_length(data_cfg):
        """ Calculate the length of the network input sample """
        # This will be used by torch summary and to initialise norm layers
        if data_cfg.srbins_type == 1:
            data_cfg.dbins = get_sampling_rate_bins_type1(data_cfg)
        if data_cfg.srbins_type == 2:
            data_cfg.dbins = get_sampling_rate_bins_type2(data_cfg)
        original_length = data_cfg.dbins[-1][1]
        # mrsampling function
        num_samples_decimated = lambda nsamp, df: int(nsamp/df)
        decimation_factor = lambda sr, nsr: int(round(sr/nsr))
        # Get decimated length
        input_length = 0
        for sidx, eidx, fs in data_cfg.dbins:
            df = decimation_factor(2048., fs)
            dnsamp = num_samples_decimated(original_length, df)
            start_idx_norm = sidx/original_length
            end_idx_norm = eidx/original_length
            # Using the normalised bins idxs, get the decimated idxs
            sidx_dec = int(start_idx_norm * dnsamp)
            eidx_dec = int(end_idx_norm * dnsamp)
            input_length += (eidx_dec - sidx_dec)

        # Correct the input length using corrupted length
        # A small portion of the data is removed during MR sampling (from either end)
        if isinstance(data_cfg.corrupted_len, int):
            input_length -= 2.*data_cfg.corrupted_len
        elif isinstance(data_cfg.corrupted_len, list):
            input_length -= sum(data_cfg.corrupted_len)
        # Set the network sample length in data_cfg
        data_cfg.network_sample_length = int(input_length)
        # Get the decimated bins
        dummy_data = np.random.rand(int(input_length))
        _, data_cfg._decimated_bins = multirate_sampling(dummy_data, data_cfg, check=True)

