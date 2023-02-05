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
import numpy as np
import pandas as pd
import torch.utils.data as D

# Class balanced splitting
from sklearn.model_selection import train_test_split

# LOCAL
from configs import *
from data_configs import *


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
        
        # cfg contains the model class (see config.py)
        cfg = eval(opts.data_config)
        return cfg
    
    
    def make_export_dir(cfg):
        # Results directory (preferred: /mnt/nnarenraju)
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
            check_file = os.path.join(check_dir, "training.hdf")
            elink_file = os.path.join(check_dir, "extlinks.hdf")
            # Copy lookup tables to export_dir
            if os.path.isfile(check_file):
                # Move the training.hdf to export_dir for pipeline
                move_location = os.path.join(export_dir, 'training.hdf')
                shutil.copy(check_file, move_location)
            if os.path.isfile(elink_file):
                # Move the extlinks.hdf to export_dir for pipeline
                move_location = os.path.join(export_dir, 'extlinks.hdf')
                shutil.copy(elink_file, move_location)
        else:
            raise FileNotFoundError(f"prepare_data.get_summary: {check_dir} not found!")
            
    
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
        lookup = list(zip(ids, paths, targets, dstype))
        train = pd.DataFrame(lookup, columns=['id', 'path', 'target', 'dstype'])
        
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
                validation_data = train.loc[train['dstype'] == 'validation'].head(num_validation)
                # Get the required number of samples from each
                folds = [(training_data['id'].values, validation_data['id'].values)]
                return (train, folds)
                
        
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
                folds = [(X_train, X_valid)]
            
            elif data_cfg.dataset == 4:
                training_data = train.loc[train['dstype'] == 'training']
                validation_data = train.loc[train['dstype'] == 'validation']
                # Get the required number of samples from each
                folds = [(training_data['id'].values, validation_data['id'].values)]
        
        return (train, folds)
    
    
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
        ttargets = train_fold['target'].values
        check_class_balance = len(ttargets[ttargets == 1])/len(ttargets)
        if check_class_balance == 0.5:
            class_sample_count = np.array([len(np.where(ttargets == t)[0]) for t in np.unique(ttargets)])
            weight = 1.0 / class_sample_count
            weights = np.array([weight[t] for t in ttargets])
            tsampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        else:
            raise ValueError('Encountered a class imbalanced training dataset!')
        
        # Create the training and validation dataset objects
        train_dataset = cfg.dataset(
                data_paths=train_fold['path'].values, targets=train_fold['target'].values,
                transforms=cfg.transforms['train'], target_transforms=cfg.transforms['target'],
                signal_only_transforms=cfg.transforms['signal'], 
                noise_only_transforms=cfg.transforms['noise'],
                training = True, cfg=cfg, data_cfg=data_cfg, store_device=cfg.store_device,
                train_device=cfg.train_device, **cfg.dataset_params)
        
        # Validation dataset
        vtargets = valid_fold['target'].values
        check_class_balance = len(vtargets[vtargets == 1])/len(vtargets)
        if check_class_balance == 0.5:
            class_sample_count = np.array([len(np.where(vtargets == t)[0]) for t in np.unique(vtargets)])
            weight = 1.0 / class_sample_count
            weights = np.array([weight[t] for t in vtargets])
            vsampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        else:
            raise ValueError('Encountered a class imbalanced validation dataset!')
        
        valid_dataset = cfg.dataset(
                data_paths=valid_fold['path'].values, targets=valid_fold['target'].values,
                transforms=cfg.transforms['test'], target_transforms=cfg.transforms['target'],
                signal_only_transforms=cfg.transforms['signal'],
                noise_only_transforms=cfg.transforms['noise'],
                training=False, cfg=cfg, data_cfg=data_cfg, store_device=cfg.store_device,
                train_device=cfg.train_device, **cfg.dataset_params)
        
        return (train_dataset, valid_dataset, tsampler, vsampler)
    
    
    def get_dataloader(cfg, train_data, valid_data, tsampler, vsampler):
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
        
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        pin_memory = cfg.pin_memory
        persistent_workers = cfg.persistent_workers
        
        # Sometimes in MAC systems, setting num_workers > 0 causes a intraop warning to appear
        # This does not seem to produce any incorrect results. However it is worrying.
        # Temporary fix (suppresses the error, but does not fix the underlying issue)
        os.environ["OMP_NUM_THREADS"] = "1"
        
        # NOTE: Use of tsampler and vsampler has been deprecated as of: January 18th, 2023
        train_loader = D.DataLoader(
            train_data, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, shuffle=True,
            prefetch_factor=cfg.prefetch_factor, persistent_workers=persistent_workers)
        
        valid_loader = D.DataLoader(
            valid_data, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
            prefetch_factor=cfg.prefetch_factor, persistent_workers=persistent_workers)
        
        return (train_loader, valid_loader)
