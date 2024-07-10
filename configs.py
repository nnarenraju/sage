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

Documentation:

[1] Using OSnet

##Architecture
model = SigmaModel

# Kernel sizes on modified OSnet (type 1)
kernel_sizes = []
kernel_sizes.append([[16, 32, 64, 128, 256], [8, 16, 32, 64, 128]])
kernel_sizes.append([[8, 16, 32, 64, 128], [2, 4, 8, 16, 32]])
kernel_sizes.append([[2, 4, 8, 16, 32], [2, 4, 8, 16, 32]])

model_params = dict(
    ## OSnet + Resnet50 CBAM
    model_name='sigmanet',
    norm_layer = 'instancenorm',
    ## OSnet params
    # channels[0] is used when initial_dim_reduction == True
    channels=[16, 32, 64, 128],
    kernel_sizes=kernel_sizes, 
    # strides[:2] is used when initial_dim_reduction == True
    strides=[2,2,8,4],
    stacking=False,
    initial_dim_reduction=False,
    # reduction value of 16 does not work with KaggleNet type kernels
    channel_gate_reduction=8,
    # ResNet CBAM params
    resnet_size = 50,
    # Common
    store_device = 'cuda:2',
)

"""

# PACKAGES
import os
import numpy as np
import torch.optim as optim

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, ReduceLROnPlateau, StepLR

# LOCAL
from data.datasets import MLMDC1, MinimalOTF
from architectures.frontend import KappaModel, ZetaModel, KappaModel_ResNet_CBAM, OmegaModel_ResNet_CBAM, KappaModel_Res2Net, SigmaModel, KappaModel_ResNet
from architectures.frontend import KappaModel_ResNet_small
from data.transforms import Unify, UnifySignal, UnifyNoise, UnifySignalGen, UnifyNoiseGen
from data.transforms import BandPass, HighPass, Whiten, MultirateSampling, Normalise, Resample, Buffer, Crop
from data.transforms import AugmentDistance, AugmentPolSky, AugmentOptimalNetworkSNR
from data.transforms import CyclicShift, AugmentPhase, Recolour
from data.transforms import GenerateWaveform, FastGenerateWaveform, GlitchAugmentGWSPY, RandomNoiseSlice, MultipleFileRandomNoiseSlice
from losses.custom_loss_functions import BCEgw_MSEtc, regularised_BCELoss, regularised_BCEWithLogitsLoss

# RayTune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


    
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
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b']),
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


class KaggleNetOTF_bigboi:
    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Feb03_bigboi"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

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
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 152,
        store_device = 'cuda:1',
    )

    """ Epochs and Batches """
    num_epochs = 500
    batch_size = 64
    save_freq = 1
    
    """ Save samples """
    num_sample_save = 100
    
    """ Weight Types """
    weight_types = ['loss', 'accuracy', 'roc_auc', 'low_far_nsignals']
    
    # Save weights for particular epochs
    save_epoch_weight = list(range(500))
    
    # Pick one of the above weights for best epoch save directory
    save_best_option = 'loss'

    # Checkpoints
    save_checkpoint = True
    checkpoint_freq = 1 # every n epochs
    resume_from_checkpoint = False
    checkpoint_path = ""
    
    pretrained = False
    weights_path = 'weights_loss.pt'
    
    # This is automatically written in export_dir
    output_loss_file = "losses.txt"
    
    """ Parameter Estimation """
    parameter_estimation = ('norm_tc', 'norm_mchirp', )

    """ Optimizer """
    ## Adam 
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)

    """ Scheduler """
    ## Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    
    # Early stoppping
    early_stopping = False

    """ Evaluation Metric """
    eval_metric = None
    # Normalised threshold for accuracy
    accuracy_thresh = 0.5
    
    """ Gradient Clipping """
    clip_norm = 10000

    """ Automatic Mixed Precision """
    # Keep this turned off when using Adam
    # It seems to be unstable and produces NaN losses
    do_AMP = False

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    """ Dataloader params """
    num_workers = 32
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.2580,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )
    
    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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

    # Do not set this to True
    # RandomNoiseSlice does the same thing but better
    batchshuffle_noise = False

    """ Optional things to do during training """
    # Plots the input to the network (including transformations) 
    # and output from the network
    network_io = False
    permitted_models = ['KappaModel', 'KappaModelPE', 'KappaModel_ResNet_CBAM', 
                        'KappaModel_Res2Net']
    # Extremes only plot
    extremes_io = False
    # Plotting on first batch
    plot_on_first_batch = False
    # Testing on a small 64000s dataset at the end of each epoch
    epoch_testing = False
    epoch_testing_dir = "/local/scratch/igr/nnarenraju/testing_64000_D4_seeded"
    epoch_far_scaling_factor = 64000.0

    """ Testing Phase """
    injection_file = 'injections.hdf'
    evaluation_output = 'evaluation.hdf'

    test_foreground_dataset = "foreground.hdf"
    test_foreground_output = "testing_foutput.hdf"
    
    test_background_dataset = "background.hdf"
    test_background_output = "testing_boutput.hdf"

    # Run device for testing phase
    ## Testing config
    # Real step will be slightly different due to rounding errors
    step_size = 0.1
    # Based on prediction probabilities in best epoch
    trigger_threshold = 0.0
    # Time shift the signal by multiple of step_size and check pred probs
    cluster_threshold = 0.0001
    # Run device for testing phase
    testing_device = 'cuda:1'

    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    far_scaling_factor = 2592000.0

    # When debug is False the following plots are not made
    # SAMPLES, DEBUG, CNN_OUTPUT
    debug = False
    debug_size = 10000

    verbose = True


class KaggleNetOTF_BOY(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "KaggleNet50_CBAM_OTF_Feb03_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'


class KaggleNetOTF_BOY_rerun(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War
    # Apr21 IMRPhenomD completed at 50 epochs. Rerun with checkpoint file from last epoch.
    name = "KaggleNet50_CBAM_IMRPhenomD_OTF_May01_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )

    """ Checkpoints """
    save_checkpoint = True
    checkpoint_freq = 1 # every n epochs
    resume_from_checkpoint = True
    checkpoint_path = "./weights/checkpoint_epoch_49_PhenomD_BOY.pt"

    """ Epochs and Batches """
    num_epochs = 500
    # Save weights for particular epochs
    save_epoch_weight = list(range(50, 500))

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'


class KaggleNetOTF_BOY_IMRPhenomD(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War
    name = "KaggleNet50_CBAM_IMRPhenomD_OTF_Apr21_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'


class KaggleNetOTF_BOY_XPHM_noHM(KaggleNetOTF_bigboi):
    # VARIANT: Using XP instead of XPHM (using XPHM with only dominant mode)

    """ Data storage """
    name = "KaggleNet50_CBAM_XPHM_noHM_OTF_May03_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'


class KaggleNetOTF_BOY_XPHM_noHM_CHEATY(KaggleNetOTF_bigboi):
    # VARIANT: Using XP instead of XPHM (using XPHM with only dominant mode)

    """ Data storage """
    name = "KaggleNet50_CBAM_XPHM_noHM_CHEATY_OTF_May03_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'


class KaggleNetOTF_BOY_Pv2(KaggleNetOTF_bigboi):
    # VARIANT: Using Pv2 instead of XPHM

    """ Data storage """
    name = "KaggleNet50_CBAM_Pv2_OTF_May03_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'


class KaggleNetOTF_BOY_XPHM_noHM_PSD_shift(KaggleNetOTF_bigboi):
    # VARIANT: Using XP instead of XPHM (using XPHM with only dominant mode)

    """ Data storage """
    name = "KaggleNet50_CBAM_XPHM_noHM_PSDshift_OTF_May04_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             use_shifted=True, shift_up_factor=10, shift_down_factor=10,
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    test_foreground_output = "testing_foutput_psd_shift.hdf"
    test_background_output = "testing_boutput_psd_shift.hdf"



### FINAL EXPERIMENTS ###

# DONE
class SageNetOTF_uTau_SNR01(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    ### SNR01 - SNR Experiment 01
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Unbounded uniformTau distribution (done)
    # 3. no PE to ease aggregation (done)
    # 4. SNR halfnorm (**VARIATION**)
    # 5. Extra O3b noise (**VARIATION**)

    """ Data storage """
    name = "SageNet50_uTau_halfnormSNR_noPE_15s_May17"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'


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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                sample_length=20.0,
                                                debug_me=False,
                                                debug_dir=""),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


# DONE (BEST 24/06/24)
class SageNetOTF_uTau_SNR02(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    ### SNR02 - SNR Experiment 02
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "SageNet50_halfnormSNR_May17_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=17.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'


# DONE (stopped very early)
class SageNetOTF_uTau_WM01(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    ### WM01 - Waveform Model Experiment 01
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Bounded uniformTau distribution (**VARIATION**)
    # 3. With PE (**VARIATION**)
    # 4. 113 days of O3b data (**VARIATION**)
    # 5. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "SageNet50_bounded_uTau_halfnormSNR_withPE_15s_Jun06"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'


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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=20.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


# DONE (stopped due to server shutdown)
class SageNetOTF_BBOY_noPSDaug(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    ### SNR02 - SNR Experiment 02
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)
    # 3. PSD augmentation turned off (**VARIATION**)

    """ Data storage """
    name = "SageNet50_halfnormSNR_noPSDaug_Jun06_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=17.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'


# RUNNING (I actually have high hopes for this one)
class SageNetOTF_metric_density(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density - fixed (**VARIATION**)

    """ Data storage """
    name = "SageNet50_metric_density_Jun26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=17.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_metric.hdf"    
    test_background_output = "testing_boutput_metric.hdf"



# ABLATION 1 - small network resnet50
class SageNetOTF_metric_lowvar(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density (**VARIATION**)
    # 4. Low variation via fixed size dataset and lack of augmentation 

    """ Data storage """
    name = "SageNet50_metric_lowvar_Jul00"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=17.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_small

    model_params = dict(
        # Resnet50
        timm_params = {'model_name': 'resnet50',
                       'pretrained': False, 
                       'in_chans': 2, 
                       'drop_rate': 0.25
                      },
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_metric_small.hdf"    
    test_background_output = "testing_boutput_metric_small.hdf"






# RUNNING (stopping prematurely, will start again later after fixing the sampling issue)
class SageNetOTF_umcq(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Uniform on p(mchirp, q) space (**VARIATION**)

    """ Data storage """
    name = "SageNet50_umcq_Jun24"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=17.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'


# RUNNING (High hopes as well)
class SageNetOTF_metric_density_noCheatyPSDaug(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density - fixed (**VARIATION**)

    """ Data storage """
    name = "SageNet50_metric_density_noCheatyPSDaug_Jun26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=17.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             use_shifted=True, shift_up_factor=10, shift_down_factor=1,
                             h1_psds_hdf="./notebooks/tmp/psds_H1_latter51days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_latter51days_20s.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'


# RUNNING
class SageNetOTF_metric_density_noCheatyPSDaug_noPSDshift(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density - fixed (**VARIATION**)

    """ Data storage """
    name = "SageNet50_metric_density_noCheatyPSDaug_noPSDshift_Jun26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=12.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=17.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 sample_length=17.0,
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    pfixed = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    # Adding a random noise realisation during the data loading process
    # Procedure should be available within dataset object
    # Fixed noise realisation method has been deprecated
    add_random_noise_realisation = True

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             use_shifted=False, shift_up_factor=10, shift_down_factor=1,
                             h1_psds_hdf="./notebooks/tmp/psds_H1_latter51days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_latter51days_20s.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'


### POTENTIAL RUNS ###
# 1. uniform on (mchirp, q)
# 2. uniform on (tau0, q)
# 3. power law on (tau0, q)
# 4. uniform + power law on (tau0, q)




### OLD RUNS ###

class KaggleNetOTF_BOY_rerun_resnet50(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "KaggleNet50_IMRPhenomD_OTF_Apr25_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet

    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        store_device = 'cuda:2',
        timm_params = {'model_name': 'resnet50', 
                       'pretrained': True, 
                       'in_chans': 2, 
                       'drop_rate': 0.25
                    }
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'


class KaggleNetOTF_BOY_rerun_res2net(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "Kaggle2Net50_CBAM_IMRPhenomD_OTF_Apr21_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_Res2Net

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

    # Run device for testing phase
    testing_device = 'cuda:2'


class KaggleNetOTF_BOY_transfer(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "KaggleNet50_CBAM_OTF_Apr14_BOY_transfer_lowSNRtrain"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
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

    # Run device for testing phase
    testing_device = 'cuda:2'

    pretrained = True
    # weights_path = './weights/weights_loss_BOY.pt'
    weights_path = 'weights_loss.pt'

    """ Optimizer """
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-6, weight_decay=1e-6)

    # Rescaling the SNR (mapped into uniform distribution)
    rescale_snr = True
    rescaled_snr_lower = 5.0
    rescaled_snr_upper = 8.0


class KaggleNetOTF_BOY_simplified(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "KaggleNet50_CBAM_OTF_Apr21_BOY_simplified_noPE"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class KaggleNetOTF_BOY_halfnormSNR(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "KaggleNet50_CBAM_halfnormSNR_OTF_Apr05_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
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

    # Run device for testing phase
    testing_device = 'cuda:1'

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class KaggleNetOTF_BOY_SNRgivenMc(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "KaggleNet50_CBAM_SNRgivenMc_OTF_Apr09_BOY"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_mc_func=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class OmegaNetOTF(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "OmegaNet50_CBAM_OTF_Feb15"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = OmegaModel_ResNet_CBAM

    model_params = dict(
        # Res2net152
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 34,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=25.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=25.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=25.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class KaggleNetOTF_longer(KaggleNetOTF_bigboi):
    ### Deviations from BOY ###
    # 1. 20 second samples instead of 12 seconds
    ## 2. Gravity Spy (O3b glitches) aug turned on (BOY in on as well)
    # 3. Using Res2Net50 instead of ResNetCBAM50
    # 4. Using ReduceLRonPlateau instead of CosineAnnealingWithWarmRestarts
    ## 5. Running for 100 epochs
    ## 6. All noise is recoloured
    ## 7. Transfer learned from "KaggleNet50_CBAM_OTF_Feb13_longer" at epoch 56 best loss

    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Feb29_longer"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_Res2Net

    model_params = dict(
        # Res2net50
        filter_size = 32,
        kernel_size = 64,
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    # PRETRAINED
    pretrained = False
    weights_path = "weights_loss.pt"

    # Save weights for particular epochs
    save_epoch_weight = list(range(100))

    """ Epochs and Batches """
    num_epochs = 100

    # Scheduler
    scheduler = ReduceLROnPlateau
    scheduler_params = dict(mode='min', factor=0.5, patience=2, min_lr=1e-6, threshold=1e-3)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=20.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=25.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=25.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.2580,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=1.0,
                             sample_length_in_s=25.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class KaggleNetOTF_CBAM_longer(KaggleNetOTF_bigboi):
    ### Deviations from BOY ###
    # 1. 20 second samples instead of 12 seconds
    # 2. Gravity Spy (O3b glitches) aug turned off
    # 3. Running for 60 epochs

    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Feb14_longer"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'

    """ Epochs and Batches """
    num_epochs = 60

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=25.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=25.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=25.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class KaggleNetOTF_betaSNR(KaggleNetOTF_bigboi):
    """ Data storage """
    # WARNING: For optimal detection sensitivity, you have to pronounce "BOY" the way Kratos does in God of War 
    name = "KaggleNet50_CBAM_OTF_Feb03_BetaSNR"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
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

    # Run device for testing phase
    testing_device = 'cuda:2'

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_beta=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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

    # Rescaling the SNR
    rescale_snr = True
    rescaled_snr_lower = 5.0
    rescaled_snr_upper = 20.0


class SigmaNetOTF(KaggleNetOTF_bigboi):
    ### Deviations from BOY ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. O3a + O3b noise realisation with all recoloured (done)
    # 3. Running for 100 epochs (done)
    # 4. Use Res2Net152 (done)
    # 5. Turn off PE, a conflicting task (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet152_Res2Net_noPE_OTF_Feb23"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_Res2Net

    model_params = dict(
        # Res2net50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 152,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

    """ Optimiser """
    #optimizer = optim.SGD
    #optimizer_params = dict(lr=1e-3, momentum=0.9, weight_decay=1e-6)

    """ Automatic Mixed Precision """
    # Keep this turned off when using Adam
    # It seems to be unstable and produces NaN losses
    #do_AMP = True

    """ Loss Function """
    # If gw_critetion is set to None, torch.nn.BCEWithLogitsLoss() is used by default
    # Extra losses cannot be used without BCEWithLogitsLoss()
    # All parameter estimation is done only using MSE loss at the moment
    loss_function = BCEgw_MSEtc(gw_criterion=None, weighted_bce_loss=False, mse_alpha=0.0,
                                emphasis_type='raw',
                                noise_emphasis=False, noise_conditions=[('min_noise', 'max_noise', 0.5),],
                                signal_emphasis=False, signal_conditions=[('min_signal', 'max_signal', 3.0),],
                                snr_loss=False, snr_conditions=[(5.0, 10.0, 0.3),],
                                mchirp_loss=False, mchirp_conditions=[(25.0, 45.0, 0.3),],
                                dchirp_conditions=[(130.0, 350.0, 1.0),],
                                variance_loss=False)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.2580,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=1.0,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_legacy(KaggleNetOTF_bigboi):
    ### Deviations from BOY ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. O3a + O3b noise realisation with all recoloured (done)
    # 3. Running for 100 epochs (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_SGDR_OTF_Feb21"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.2580,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=1.0,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_uTau(KaggleNetOTF_bigboi):
    ### Deviations from BOY ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. UniformTau distribution

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_SGDR_uTau_OTF_Mar03"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.2580,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_uTau_simplified(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. UniformTau distribution (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_uTau_OTF_Mar16"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_bounded_uTau_simplified(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. UniformTau distribution (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)
    # 3. Bounded uTau distribution (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_bounded_uTau_OTF_Mar19"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    weights_path = 'weights_low_far_nsignals_49.pt'
    # Run device for testing phase
    testing_device = 'cuda:1'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_bounded_plMc_simplified(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. UniformTau distribution (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)
    # 3. Bounded power law mchirp distribution (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_bounded_plMc_OTF_Mar28"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_bounded_plTau_add5SNR(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. UniformTau distribution (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)
    # 3. Bounded power law mchirp distribution (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_bounded_plTau_add5SNR_OTF_Apr03"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

    """ Gradient Clipping """
    clip_norm = 1

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_add5=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_bounded_uMc(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. UniformTau distribution (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)
    # 3. Bounded uMc distribution (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_bounded_uMc_OTF_Mar24"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_uTau_simplified_add5SNR(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. UniformTau distribution (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)
    # 3. SNR rescaling = SNR_old + 5

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_uTau_add5SNR_OTF_Mar16"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_add5=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_simplified_add5SNR(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)
    # 3. SNR rescaling = SNR_old + 5 (done)
    # 4. Running on testing distribution (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_add5SNR_OTF_Apr02"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_add5=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_uMc_simplified_add5SNR(KaggleNetOTF_bigboi):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    ### Secondary Deviations (Comparison to uTau) ###
    # 1. no PE to ease aggregation (done)
    # 2. Turn off O3b glitches (done)
    # 3. SNR rescaling = SNR_old + 5 (done)
    # 4. Running on testing distribution (done)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_uMc_add5SNR_OTF_Apr02"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

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

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.0,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_add5=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days_20s.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days_20s.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_uMc(KaggleNetOTF_bigboi):
    ### Deviations from BOY ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. Uniform chirp mass distribution

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_SGDR_uMc_OTF_Mar03"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.2580,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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


class SigmaNetOTF_uMc_bounded(KaggleNetOTF_bigboi):
    ### Deviations from BOY ###
    # 1. 15 second sample instead of 12 seconds (done)
    # 2. Running for 100 epochs (done)
    # 3. Uniform chirp mass distribution with bounded m1, m2 (CDF snake)

    ## Sigma indicating the sum of all my hard work
    ## This will be the final model published on the ORChiD paper

    """ Data storage """
    name = "SigmaNet50_CBAM_ubMc_OTF_Mar09"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()
    
    """ Architecture """
    model = KappaModel_ResNet_CBAM

    model_params = dict(
        # Resnet50 CBAM
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:1',
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:2'

    test_foreground_output = "testing_foutput_uMcb.hdf"
    test_background_output = "testing_boutput_uMcb.hdf"

    """ Epochs and Batches """
    num_epochs = 100

    # Save weights for particular epochs
    save_epoch_weight = np.arange(100)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(sample_length=15.0),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_training')
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    sample_length=20.0, segment_llimit=0, segment_ulimit=132, debug_me=False,
                                    debug_dir=os.path.join(debug_dir, 'RandomNoiseSlice_validation')
                                ),
                    },
                    GlitchAugmentGWSPY(include=['H1_O3b', 'L1_O3b'], debug_me=False,
                                       debug_dir=os.path.join(debug_dir, 'GravitySpy')),
                    pfixed = 0.2580,
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf="./notebooks/tmp/psds_H1_30days.hdf",
                             l1_psds_hdf="./notebooks/tmp/psds_L1_30days.hdf",
                             p_recolour=0.3829,
                             sample_length_in_s=20.0,
                             debug_me=False,
                             debug_dir=os.path.join(debug_dir, 'Recolour')),
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
