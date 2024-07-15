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
import subprocess
import numpy as np
import torch.optim as optim

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# LOCAL
from data.datasets import MinimalOTF
from architectures.frontend import KappaModel_ResNet_CBAM
from architectures.frontend import KappaModel_ResNet_small
from data.transforms import Unify, UnifySignal, UnifyNoise, UnifySignalGen, UnifyNoiseGen
from data.transforms import Whiten, MultirateSampling, Normalise
from data.transforms import AugmentOptimalNetworkSNR
from data.transforms import Recolour
from data.transforms import FastGenerateWaveform, GlitchAugmentGWSPY, RandomNoiseSlice, MultipleFileRandomNoiseSlice
from losses.custom_loss_functions import BCEgw_MSEtc

# RayTune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


    
""" CUSTOM MODELS FOR EXPERIMENTATION """

class KaggleNetOTF_bigboi:
    
    """ Data storage """
    name = "KaggleNet50_CBAM_OTF_Feb03_bigboi"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/DEBUGGING") / name
    debug_dir = "./DEBUG"
    repo_abspath = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)

    """ RayTune """
    # Placed before initialising any relevant tunable parameter
    # WARNING: Required compute is prohibitively large for large models
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
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
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


### FINAL EXPERIMENTS ###

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
    test_foreground_output = "testing_foutput_metric_latest.hdf"    
    test_background_output = "testing_boutput_metric_latest.hdf"


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

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_noCheaty.hdf"    
    test_background_output = "testing_boutput_noCheaty.hdf"


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
        store_device = 'cuda:0',
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_noCheaty_noShift.hdf"    
    test_background_output = "testing_boutput_noCheaty_noShift.hdf"


### POTENTIAL RUNS ###
# 1. uniform on (mchirp, q)
# 2. uniform on (tau0, q)
# 3. power law on (tau0, q)
# 4. uniform + power law on (tau0, q)