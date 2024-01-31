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
import numpy as np
import torch.optim as optim

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, ReduceLROnPlateau, StepLR

# LOCAL
from data.datasets import MLMDC1, MinimalOTF
from architectures.frontend import GammaModel, KappaModel, ZetaModel, KappaModel_ResNet_CBAM, IotaModelPE
from data.transforms import Unify, UnifySignal, UnifyNoise, UnifySignalGen, UnifyNoiseGen
from data.transforms import BandPass, HighPass, Whiten, MultirateSampling, Normalise, Resample, Buffer, Crop
from data.transforms import AugmentDistance, AugmentPolSky, AugmentOptimalNetworkSNR
from data.transforms import CyclicShift, AugmentPhase, Recolour
from data.transforms import GenerateWaveform, FastGenerateWaveform, GlitchAugmentGWSPY, RandomNoiseSlice
from losses.custom_loss_functions import BCEgw_MSEtc, regularised_BCELoss, regularised_BCEWithLogitsLoss

# RayTune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



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
    
    """ Weight Types """
    weight_types = ['loss', 'accuracy', 'roc_auc', 'lmax_noise_stat', 'lmin_noise_stat',
                    'hmax_signal_stat', 'hmin_signal_stat', 'best_noise_stat', 'best_signal_stat',
                    'best_stat_compromise', 'best_overlap_area', 'best_signal_area', 'best_noise_area',
                    'best_diff_distance']
    
    # Pick one of the above weights for best epoch save directory
    save_best_option = 'loss'
    
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


class KaggleFirst_REF(KaggleFirst):
    
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
    

class KaggleFirstPE_BASELINE(KaggleFirst_REF):
    
    """ Data storage """
    name = "KaggleNet50_CheckRealNoise"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    save_remarks = 'Training-D4'
    
    """ RayTune """
    # Placed before initialising any relevant tunable parameter
    rtune_optimise = False
    
    rtune_params = dict(
        # RayTune Tunable Parameters
        config = {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([32,])
        },
        # Scheduler (ASHA has intelligent early stopping)
        scheduler = ASHAScheduler,
        # NOTE: max_t is maximum number of epochs Tune is allowed to run
        scheduler_params = dict(
            metric = "loss",
            mode = "min",
            max_t = 10,
            grace_period = 1,
            reduction_factor = 2
        ),
        # Reporter
        reporter = CLIReporter,
        reporter_params = dict(
            # parameter_columns=["l1", "l2", "lr", "batch_size"],
            metric_columns=["loss", "accuracy", "low_far_nsignals", "training_iteration"]
        ),
        # To sample multiple times/run multiple trials
        # Samples from config num_samples number of times
        num_samples = 100
    )

    """ Weights and Biases (Wandb) """
    use_wandb_logging = False
    
    """ Dataset """
    dataset = MLMDC1
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM
    
    model_params = dict(
        # Res2net50
        filter_size = 32,
        kernel_size = 64,
        store_device = 'cuda:1',
    )

    """ Epochs and Batches """
    num_epochs = 50
    batch_size = 64
    save_freq = 1
    
    """ Save samples """
    num_sample_save = 100
    
    """ Weight Types """
    weight_types = ['loss', 'accuracy', 'roc_auc', 'low_far_nsignals']
    
    # Save weights for particular epochs
    save_epoch_weight = list(range(50))
    
    # Pick one of the above weights for best epoch save directory
    save_best_option = 'loss'
    
    pretrained = False
    weights_path = 'weights_loss.pt'
    
    """ Parameter Estimation """
    parameter_estimation = ('norm_tc', 'norm_mchirp', )
    
    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'
    
    """ Optimizer """
    ## Stochastic Gradient Descent
    #optimizer = optim.SGD
    #optimizer_params = dict(lr=1e-3, momentum=0.9, weight_decay=1e-6)
    ## Adam 
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
    ## AdamW
    #optimizer = optim.AdamW
    #optimizer_params = dict(lr=2e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

    """ Scheduler """
    ## Lambda LR
    # Default option: scheduler = LambdaLR
    # lambda1 = lambda epoch: 0.95 ** epoch
    ## ReduceLR on Plateau
    # scheduler = ReduceLROnPlateau
    # scheduler_params = dict(mode='min', factor=0.5, patience=3, min_lr=1e-5)
    ## Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    #scheduler = StepLR
    #scheduler_params = dict(step_size=2, gamma=0.7)
    
    """ Gradient Clipping """
    clip_norm = 10000

    """ Automatic Mixed Precision """
    # Keep this turned off when using Adam
    # It seems to be unstable and produces NaN losses
    do_AMP = False
    
    """ Loss Function """
    # If gw_critetion is set to None, torch.nn.BCEWithLogitsLoss() is used by default
    # Extra losses cannot be used without BCEWithLogitsLoss()
    # All parameter estimation is done only using MSE loss at the moment
    loss_function = BCEgw_MSEtc(gw_criterion=None, weighted_bce_loss=False, mse_alpha=1.0,
                                emphasis_type='raw',
                                noise_emphasis=False, noise_conditions=[('min_noise', 'max_noise', 0.5),],
                                signal_emphasis=False, signal_conditions=[('min_signal', 'max_signal', 1.0),],
                                snr_loss=False, snr_conditions=[(5.0, 10.0, 0.3),],
                                mchirp_loss=False, mchirp_conditions=[(25.0, 45.0, 0.3),],
                                dchirp_conditions=[(130.0, 350.0, 1.0),],
                                variance_loss=False)

    # These params must be present in target dict
    # Make sure that params of PE are also included within this (generalise this!)
    weighted_bce_loss_params = ('mchirp', 'tc', 'q', )
    
    # Rescaling the SNR (mapped into uniform distribution)
    rescale_snr = True
    rescaled_snr_lower = 5.0
    rescaled_snr_upper = 15.0
    
    # Calculate the network SNR for pure noise samples as well
    # If used with parameter estimation, custom loss function should have network_snr_for_noise option toggled
    network_snr_for_noise = False

    # Dataset imbalance
    ignore_dset_imbalance = False
    subset_for_funsies = False # debug_size is used for subset, debug need not be true
    
    """ Dataloader params """
    num_workers = 32
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True

    """ Generation """
    generation = dict(
        signal = None,
        noise = None 
    )

    """ Transforms """    
    transforms = dict(
        signal=UnifySignal([
                    AugmentPolSky(),
                    AugmentOptimalNetworkSNR(rescale=True),
                ]),
        noise=None,
        train=Unify({
                    'stage1':[
                            HighPass(lower=20, fs=2048., order=10),
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=True),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            HighPass(lower=20, fs=2048., order=10),
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=True),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )
    
    batchshuffle_noise = True
    
    """ Optional things to do during training """
    # Plots the input to the network (including transformations) 
    # and output from the network
    network_io = False
    permitted_models = ['KappaModel', 'KappaModelPE', 'KappaModel_ResNet_CBAM']
    # Bad high SNR signals plotting
    bad_snr_stat_thresh = 8.0 # less than this value in network output
    high_snr_thresh = 12.0 # greater than this value in source network SNR
    # Extremes only plot
    extremes_io = False
    # Plotting on first batch
    plot_on_first_batch = False
    # Testing on a small 32000s dataset at the end of each epoch
    epoch_testing = False
    epoch_testing_dir = "/local/scratch/igr/nnarenraju/testing_64000_D4_seeded"
    epoch_far_scaling_factor = 64000.0
    
    """ Testing Phase """
    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    injection_file = 'injections.hdf'
    evaluation_output = 'evaluation.hdf'
    # FAR scaling factor --> seconds per month
    # Short: far_scaling_factor = 64000.0
    # Month: far_scaling_factor = 2592000.0
    far_scaling_factor = 2592000.0
    
    test_foreground_dataset = "foreground.hdf"
    test_foreground_output = "testing_foutput.hdf"
    
    test_background_dataset = "background.hdf"
    test_background_output = "testing_boutput.hdf"
    
    ## Testing config
    # Real step will be slightly different due to rounding errors
    step_size = 0.1
    # Based on prediction probabilities in best epoch
    trigger_threshold = 0.0
    # Time shift the signal by multiple of step_size and check pred probs
    cluster_threshold = 0.0001
    # Run device for testing phase
    testing_device = 'cuda:0'
    
    # When debug is False the following plots are not made
    # SAMPLES, DEBUG, CNN_OUTPUT
    debug = False
    debug_size = 10000

    verbose = True


class VirgoNet_Dec01(KaggleFirstPE_BASELINE):

    """ Data storage """
    name = "VirgoNet_Dec01_Adam_CAWR5_CombinationDset"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Architecture """
    model = ZetaModel

    model_params = dict(
        store_device = 'cuda:1',
    )
     
    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1' 

    """ Loss Function """
    # If gw_critetion is set to None, torch.nn.BCEWithLogitsLoss() is used by default
    # Extra losses cannot be used without BCEWithLogitsLoss()
    # All parameter estimation is done only using MSE loss at the moment
    loss_function = BCEgw_MSEtc(gw_criterion=None, weighted_bce_loss=False, mse_alpha=0.0,
                                emphasis_type='raw',
                                noise_emphasis=False, noise_conditions=[('min_noise', 'max_noise', 0.5),],
                                signal_emphasis=False, signal_conditions=[('min_signal', 'max_signal', 1.0),],
                                snr_loss=False, snr_conditions=[(5.0, 10.0, 0.3),],
                                mchirp_loss=False, mchirp_conditions=[(25.0, 45.0, 0.3),],
                                dchirp_conditions=[(130.0, 350.0, 1.0),],
                                variance_loss=False)


class KaggleNetOTF_BASELINE(KaggleFirstPE_BASELINE):
    # OTF automatically includes OTF signal generation and random noise slice for noise

    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Jan30_Adam_morenoise_uniform_mchirp"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Architecture """
    model = KappaModel_ResNet_CBAM
    model_params = dict(
        # Res2net50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )

    """ Dataset """
    dataset = MinimalOTF
    
    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    """ Dataloader params """
    num_workers = 16

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=-1
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132
                                ),
                    },
                    GlitchAugmentGWSPY(),
                    pfixed = 0.33
                )
    )
    
    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True),
                ]),
        noise=None,
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    # Do not set this to True
    # RandomNoiseSlice does the same thing but better
    batchshuffle_noise = False

    """ Optional things to do during training """
    # Plots the input to the network (including transformations) 
    # and output from the network
    network_io = False
    permitted_models = ['KappaModel', 'KappaModelPE', 'KappaModel_ResNet_CBAM', 'KappaModel_Res2Net']
    # Extremes only plot
    extremes_io = False
    # Plotting on first batch
    plot_on_first_batch = False
    # Testing on a small 64000s dataset at the end of each epoch
    epoch_testing = False
    epoch_testing_dir = "/local/scratch/igr/nnarenraju/testing_64000_D4_seeded"
    epoch_far_scaling_factor = 64000.0

    """ Testing Phase """
    # Run device for testing phase
    weights_path = 'weights_loss.pt'
    testing_device = 'cuda:2'
    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    far_scaling_factor = 2592000.0


class KaggleNetOTF_PSDaug(KaggleNetOTF_BASELINE):
    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Jan28_AugmentedPSDs"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM
    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    """ Dataloader params """
    num_workers = 16

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132
                                ),
                    },
                    GlitchAugmentGWSPY(),
                    pfixed = 0.0
                )
    )
    
    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_h1_81days.hdf", 
                             l1_psds_hdf="./notebooks/tmp/psds_l1_81days.hdf"),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    """ Testing Phase """
    # Run device for testing phase
    weights_path = 'weights_loss.pt'
    testing_device = 'cuda:0'
    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    far_scaling_factor = 2592000.0


class KaggleNetOTF_bigboi(KaggleNetOTF_BASELINE):
    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Jan30_bigboi"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM
    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 152,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    """ Dataloader params """
    num_workers = 16

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132
                                ),
                    },
                    None,
                    pfixed = 0.0
                )
    )
    
    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_h1_81days.hdf", 
                             l1_psds_hdf="./notebooks/tmp/psds_l1_81days.hdf"),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    """ Testing Phase """
    # Run device for testing phase
    weights_path = 'weights_loss.pt'
    testing_device = 'cuda:0'
    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    far_scaling_factor = 2592000.0


class KaggleNetOTF_bigboi_uniformMchirp(KaggleNetOTF_BASELINE):
    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Jan30_bigboi_uniformMchirp"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM
    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 152,
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    """ Dataloader params """
    num_workers = 16

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132
                                ),
                    },
                    None,
                    pfixed = 0.0
                )
    )
    
    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_h1_81days.hdf", 
                             l1_psds_hdf="./notebooks/tmp/psds_l1_81days.hdf"),
                ]),
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=True, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    """ Testing Phase """
    # Run device for testing phase
    weights_path = 'weights_loss.pt'
    testing_device = 'cuda:0'
    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    far_scaling_factor = 2592000.0


class InceptionNetOTF(KaggleNetOTF_BASELINE):

    """ Data storage """
    name = "InceptionNet_OTF_Jan22"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Architecture """
    model = IotaModelPE
    
    model_params = dict(
        # InceptionNet
        store_device = 'cuda:2',
    )
    
    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    """ Dataloader params """
    num_workers = 16

    """ Optional things to do during training """
    # Plots the input to the network (including transformations) 
    # and output from the network
    network_io = False
    permitted_models = ['KappaModel', 'KappaModelPE', 'KappaModel_ResNet_CBAM', 'KappaModel_Res2Net']
    # Extremes only plot
    extremes_io = False
    # Plotting on first batch
    plot_on_first_batch = False

    """ Testing Phase """
    # Run device for testing phase
    weights_path = 'weights_loss.pt'
    testing_device = 'cuda:0'
    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    far_scaling_factor = 2592000.0


class VirgoNetOTF(KaggleNetOTF_BASELINE):

    """ Data storage """
    name = "VirgoNet_OTF_Jan12_Adam"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Architecture """
    model = ZetaModel

    model_params = dict(
        store_device = 'cuda:0',
    )
     
    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0' 

    """ Loss Function """
    # If gw_critetion is set to None, torch.nn.BCEWithLogitsLoss() is used by default
    # Extra losses cannot be used without BCEWithLogitsLoss()
    # All parameter estimation is done only using MSE loss at the moment
    loss_function = BCEgw_MSEtc(gw_criterion=None, weighted_bce_loss=False, mse_alpha=0.0,
                                emphasis_type='raw',
                                noise_emphasis=False, noise_conditions=[('min_noise', 'max_noise', 0.5),],
                                signal_emphasis=False, signal_conditions=[('min_signal', 'max_signal', 1.0),],
                                snr_loss=False, snr_conditions=[(5.0, 10.0, 0.3),],
                                mchirp_loss=False, mchirp_conditions=[(25.0, 45.0, 0.3),],
                                dchirp_conditions=[(130.0, 350.0, 1.0),],
                                variance_loss=False)