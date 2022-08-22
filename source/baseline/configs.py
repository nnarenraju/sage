# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Nov 27 17:09:58 2021

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
import json
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# LOCAL
from data.datasets import MLMDC1
from architectures.frontend import GammaModel, KappaModel, KappaModelPE
from data.transforms import Unify, UnifySignal, UnifyNoise
from data.transforms import BandPass, HighPass, Whiten, MultirateSampling
from data.transforms import AugmentDistance, AugmentPolSky, CyclicShift
from losses.custom_loss_functions import BCEgw_MSEtc, regularised_BCELoss, regularised_BCEWithLogitsLoss


""" DEFAULT (Lightning Baseline)"""

class Baseline:
    
    """ Data storage """
    name = "Baseline_longer"
    # Directory to store output from pipeline/lightning
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    
    """ Dataset Splitting """
    # Number of folds (must be at least 2, default = 5)
    n_splits = 2
    # Seed for K-Fold shuffling
    seed = 42
    # Folds are made by preserving the percentage of samples for each class
    # If set to None, dataset is split into 80-20 ratio for training and validation by default
    splitter = None
    
    """ Dataset """
    # Dataset object (opts, quick access, read only)
    # if dataset object is set to 'simple', trainable.hdf should exist
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = GammaModel
    
    model_params = dict(
        # Simple NN Model
        model_name = 'mlmdc_example',
        in_channels = 2,
        out_channels = 2,
        store_device = 'cpu',
    )
    
    # Pre-trained weights (does not refer to pretrained timm models)
    # This refers to a pretrained frontend+backend model on GW data
    pretrained = False
    # Provide a weights path even if pretrained==False
    # This location will be used to store the best weights
    # Just the file_name is enough, automagically stored in export_dir
    weights_path = 'weights.pt'
    
    """ Save trainable train and valid data """
    # Saves trainable data after transforms are applied as HDF5 files
    # This can then be used alongside a simple DataLoader
    # Set batch_size to 1, so each sample is stored separately (unless chunks are req.)
    ## WARNING: This may take a while. Check storage space before continuing.
    save_trainable_dataset = False
    
    """ Epochs and Batches """
    num_steps = 25000
    num_epochs = 25
    # Equal batch_size for both training and validation
    batch_size = 100
    # every 'n' epochs
    save_freq = 5
    # Overfitting check and Early Stopping
    early_stopping = True
    
    """ DataLoader params """
    num_workers = 0
    pin_memory = False
    
    """ Gradient Clipping """
    # Clip gradients to make convergence somewhat easier
    clip_norm = 100
    
    """ Optimizer """
    optimizer = optim.Adam
    optimizer_params = dict(lr=5e-5, weight_decay=1e-6)
    
    """ Scheduler """
    scheduler = None
    scheduler_params = dict()
    scheduler_target = None
    batch_scheduler = False
    
    """ Loss Function """
    # Add the () as suffix to loss function, returns object instead
    loss_function = regularised_BCELoss(dim=2)
    # This is automatically written in export_dir
    output_loss_file = "losses.txt"
    
    """ Evaluation Metric """
    eval_metric = None
    # Normalised threshold for accuracy
    accuracy_thresh = 0.5
    
    """ Storage Devices """
    store_device = 'cpu'
    train_device = 'cpu'
    
    """ Data Transforms """
    # Input to Unfy should always be a list
    # Input to transforms should be a dict
    # The following transforms are performed in the given order
    transforms = dict(
        train=Unify([
            BandPass(lower=16, upper=512, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        test=Unify([
            BandPass(lower=16, upper=512, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        target=None
    )
    
    # Debugging (size: train_data = 1e4, val_data = 1e3)
    debug = False
    debug_size = 10000
    # Verbosity (Not properly implemented yet. Use logger.)
    verbose = False
    
    
    
""" CUSTOM MODELS FOR EXPERIMENTATION """

class KaggleFirst:
    
    """ Data storage """
    name = "KaggleFirst"
    export_dir = Path("/Users/nnarenraju/Desktop") / name
    online_workspace = "/data/www.astro/nnarenraju"
    save_remarks = ''
    
    """ Dataset Splitting """
    n_splits = 2
    seed = 42
    splitter = None
    
    """ Dataset """
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel
    
    model_params = dict(
        # Kaggle frontend+backend
        # This model is ridiculously slow on cpu, use cuda:0
        model_name = 'kaggle_first', 
        filter_size = 16,
        kernel_size = 32,
        timm_params = {'model_name': 'resnet34', 
                        'pretrained': True, 
                        'in_chans': 2, 
                        'drop_rate': 0.25},
        store_device = 'cpu',
    )
    
    pretrained = False
    weights_path = 'weights.pt'
    
    """ Parameter Estimation """
    parameter_estimation = ()
    
    """ Epochs and Batches """
    num_epochs = 25
    batch_size = 100
    save_freq = 5
    num_sample_save = 100
    early_stopping = False
    
    """ Dataloader params """
    num_workers = 0
    pin_memory = False
    prefetch_factor = 2
    persistent_workers = False
    
    """ Gradient Clipping """
    clip_norm = 100
    
    """ Optimizer """
    optimizer = optim.Adam
    optimizer_params = dict(lr=5e-5, weight_decay=1e-6)
    
    """ Scheduler """
    scheduler = None
    scheduler_params = dict()
    
    """ Loss Function """
    loss_function = regularised_BCELoss(dim=1)
    output_loss_file = "losses.txt"
    
    """ Evaluation Metric """
    eval_metric = None
    accuracy_thresh = 0.5
    
    """ Storage Devices """
    store_device = 'cpu'
    train_device = 'cpu'
    
    """ Data Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True
    
    # Rescaling the SNR (mapped into uniform distribution)
    rescale_snr = False
    rescaled_snr_lower = 5.0
    rescaled_snr_upper = 20.0
    
    # Calculate the network SNR for pure noise samples as well
    # If used with parameter estimation, loss functions will include SNR for noise as well
    network_snr_for_noise = False
    
    transforms = dict(
        signal=UnifySignal([
            AugmentPolSky(),
            AugmentDistance(),
        ]),
        noise=UnifyNoise([
            CyclicShift(),
        ]),
        train=Unify([
            HighPass(lower=16, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        test=Unify([
            HighPass(lower=16, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
            MultirateSampling(),
        ]),
        target=None
    )
    
    """ Testing Phase """
    testing_dir = "/local/scratch/igr/nnarenraju"
    injection_file = 'testing_injections.hdf'
    evaluation_output = 'evaluation.hdf'
    # FAR scaling factor --> seconds per month
    far_scaling_factor = 30 * 24 * 60 * 60
    
    
    test_foreground_dataset = "testing_foreground.hdf"
    test_foreground_output = "testing_foutput.hdf"
    
    test_background_dataset = "testing_background.hdf"
    test_background_output = "testing_boutput.hdf"
    
    
    ## Testing config
    # Real step will be slightly different due to rounding errors
    step_size = 0.1
    # Based on prediction probabilities in best epoch
    trigger_threshold = 0.2
    # Time shift the signal by multiple of step_size and check pred probs
    cluster_threshold = 0.35
    # Run device for testing phase
    testing_device = 'cpu'
    
    """ Pipeline debug mode """
    debug = False
    debug_size = 10000
    
    """ Pipeline verbosity """
    verbose = False
    

class Baseline_May18(KaggleFirst):
    
    """ Data storage """
    name = "Baseline_Aug20"
    export_dir = Path("/home/nnarenraju/Research") / name
    save_remarks = 'Improve-Sensitivity'
    
    """ Dataset """
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = GammaModel
    
    model_params = dict(
        # Simple NN Model
        model_name = 'baseline',
        in_channels = 2,
        out_channels = 1,
        flatten_size = 464,
        store_device = 'cuda:1',
    )
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 100
    persistent_workers = True
    
    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'
    
    """ Epochs and Batches """
    num_epochs = 25
    batch_size = 32
    save_freq = 1
    
    # Rescaling the SNR (mapped into uniform distribution)
    rescale_snr = True
    rescaled_snr_lower = 0.01
    rescaled_snr_upper = 25.0
    
    """ Loss Function """
    loss_function = regularised_BCEWithLogitsLoss(dim=1)
    
    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
            AugmentPolSky(),
            AugmentDistance(),
        ]),
        noise=UnifyNoise([
            CyclicShift(),
        ]),
        train=Unify([
            HighPass(lower=16, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
        ]),
        test=Unify([
            HighPass(lower=16, fs=2048., order=6),
            Whiten(trunc_method='hann', remove_corrupted=True),
        ]),
        target=None
    )
    
    
    """ Testing Phase """
    testing_dir = "/local/scratch/igr/nnarenraju/testing_month"
    injection_file = 'injections.hdf'
    evaluation_output = 'evaluation.hdf'
    # FAR scaling factor --> seconds per month
    far_scaling_factor = 30 * 24 * 60 * 60
    
    
    test_foreground_dataset = "foreground.hdf"
    test_foreground_output = "testing_foutput.hdf"
    
    test_background_dataset = "background.hdf"
    test_background_output = "testing_boutput.hdf"
    
    ## Testing config
    # Real step will be slightly different due to rounding errors
    step_size = 0.1
    # Based on prediction probabilities in best epoch
    trigger_threshold = 0.2
    # Time shift the signal by multiple of step_size and check pred probs
    cluster_threshold = 0.35
    # Run device for testing phase
    testing_device = 'cuda:1'
    
    debug = True
    debug_size = 10000
    
    verbose = True


class KaggleFirst_Jun9(KaggleFirst):
    
    """ Data storage """
    name = "KaggleFirst_Jul11"
    export_dir = Path("/home/nnarenraju/Research") / name
    save_remarks = 'OverFitFix'
    
    """ Dataset """
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel
    
    model_params = dict(
        # Kaggle frontend+backend
        # This model is ridiculously slow on cpu, use cuda:0
        model_name = 'KaggleFirstJun9', 
        filter_size = 32,
        kernel_size = 64,
        timm_params = {'model_name': 'resnet34', 
                        'pretrained': True, 
                        'in_chans': 2, 
                        'drop_rate': 0.25},
        store_device = 'cuda:1',
    )
    
    """ Scheduler """
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 100
    persistent_workers = True
    
    """ Epochs and Batches """
    num_epochs = 10
    batch_size = 100
    save_freq = 1
    
    """ Save samples """
    num_sample_save = 100
    
    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'
    
    """ Loss Function """
    loss_function = regularised_BCEWithLogitsLoss(dim=1)
    
    """ Testing Phase """
    testing_dir = "/local/scratch/igr/nnarenraju"
    injection_file = 'testing_injections.hdf'
    evaluation_output = 'evaluation.hdf'
    # FAR scaling factor --> seconds per month
    far_scaling_factor = 30 * 24 * 60 * 60
    
    
    test_foreground_dataset = "testing_foreground.hdf"
    test_foreground_output = "testing_foutput.hdf"
    
    test_background_dataset = "testing_background.hdf"
    test_background_output = "testing_boutput.hdf"
    
    ## Testing config
    # Real step will be slightly different due to rounding errors
    step_size = 0.1
    # Based on prediction probabilities in best epoch
    trigger_threshold = 0.2
    # Time shift the signal by multiple of step_size and check pred probs
    cluster_threshold = 0.35
    # Run device for testing phase
    testing_device = 'cuda:1'
    
    debug = True
    debug_size = 1000
    

class KaggleFirstPE_Jun9(KaggleFirst_Jun9):
    
    """ Data storage """
    name = "KaggleFirst_Aug11"
    export_dir = Path("/home/nnarenraju/Research") / name
    save_remarks = 'UniformSNR-batch1000-PE-all'
    
    """ Dataset """
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModelPE
    
    model_params = dict(
        # Kaggle frontend+backend
        # This model is ridiculously slow on cpu, use cuda:0
        model_name = 'KaggleFirstPEJun9', 
        filter_size = 32,
        kernel_size = 64,
        timm_params = {'model_name': 'resnet34', 
                        'pretrained': True, 
                        'in_chans': 2, 
                        'drop_rate': 0.25},
        store_device = 'cuda:0',
    )
    
    """ Epochs and Batches """
    num_epochs = 20
    batch_size = 1000
    save_freq = 1
    
    """ Save samples """
    num_sample_save = 100
    
    """ Parameter Estimation """
    parameter_estimation = ('norm_tc', 'norm_dist', 'norm_q', 'norm_dchirp', 'norm_mchirp', 'snr', )
    
    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'
    
    """ Loss Function """
    # If gw_critetion is set to None, torch.nn.BCEWithLogitsLoss() is used by default
    # All parameter estimation is done only using MSE loss at the moment
    loss_function = BCEgw_MSEtc(mse_alpha=5.0, network_snr_for_noise=False, gw_criterion=None)
    
    # Rescaling the SNR (mapped into uniform distribution)
    rescale_snr = False
    rescaled_snr_lower = 7.0
    rescaled_snr_upper = 20.0
    
    # Calculate the network SNR for pure noise samples as well
    # If used with parameter estimation, custom loss function should have network_snr_for_noise option toggled
    network_snr_for_noise = False
    
    """ Testing Phase """
    testing_dir = "/local/scratch/igr/nnarenraju"
    injection_file = 'testing_injections.hdf'
    evaluation_output = 'evaluation.hdf'
    # FAR scaling factor --> seconds per month
    far_scaling_factor = 30 * 24 * 60 * 60
    
    
    test_foreground_dataset = "testing_foreground.hdf"
    test_foreground_output = "testing_foutput.hdf"
    
    test_background_dataset = "testing_background.hdf"
    test_background_output = "testing_boutput.hdf"
    
    ## Testing config
    # Real step will be slightly different due to rounding errors
    step_size = 0.1
    # Based on prediction probabilities in best epoch
    trigger_threshold = 0.2
    # Time shift the signal by multiple of step_size and check pred probs
    cluster_threshold = 0.35
    # Run device for testing phase
    testing_device = 'cuda:0'
    
    # When debug is False the following plots are not made
    # SAMPLES, DEBUG, CNN_OUTPUT
    debug = False
    debug_size = 10000
    
    verbose = True



""" Load config file into class """

class Imported:
    
    def __init__(self, imported_config):
        with open(imported_config) as fp:
            cfg = json.load(fp)
        
        """ Data storage """
        self.name = cfg['name']
        self.export_dir = Path(cfg['export_dir'])
        
        """ Dataset Splitting """
        self.n_splits = 2
        self.seed = 42
        self.splitter = None
        
        """ Dataset """
        self.dataset = MLMDC1
        self.dataset_params = dict()
        
        raise NotImplementedError('Imported class under construction!')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
