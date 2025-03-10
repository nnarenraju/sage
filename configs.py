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
import torch
import subprocess
import numpy as np
import torch.optim as optim

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

## LOCAL
# Dataset objects
from data.datasets import MinimalOTF

# Architectures
from architectures.models import Rigatoni_MS_ResNetCBAM, Rigatoni_MS_ResNetCBAM_legacy, Rigatoni_MS_ResNetCBAM_legacy_minimal
from architectures.models import KappaModel_ResNet1D, KappaModelPE
from architectures.frontend import MultiScaleBlock

# Transforms, augmentation and generation
from data.transforms import Unify, UnifySignal, UnifyNoise, UnifySignalGen, UnifyNoiseGen
from data.transforms import Whiten, MultirateSampling, Normalise, MonorateSampling
from data.transforms import AugmentOptimalNetworkSNR, AugmentPolSky
from data.transforms import Recolour, HighPass
from data.transforms import Buffer, BufferPerChannel
# Generating signals and noise
from data.transforms import FastGenerateWaveform, SinusoidGenerator
from data.transforms import RandomNoiseSlice, MultipleFileRandomNoiseSlice, ColouredNoiseGenerator, WhiteNoiseGenerator

# Loss functions
from losses.custom_loss_functions import BCEWithPEregLoss, lPOPWithPEregLoss

# RayTune
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# TASKS
# Cleanup prior modifications
# Change all mention on ORChiD to Sage
# All tmp files must be placed in a single location (eg. segments.csv)
# code to produce all tmp files must be consolidated (add to utils)
# code to download noise files for full experimentation must be consolidated (add to utils)
# Change all debug folders to exist within export_dir
# Clean unify noise gen
# Move all external data into one directory (psds, O3 noise, etc.)
# Add verbosity to all modules

# Add logging to all modules
# Add documentation to all classes and functions
# Add diagnostic tests with at least 90% coverage
# Add Sage logo to output

    
""" CUSTOM MODELS FOR EXPERIMENTATION """

class SageNetOTF:
    
    """ Data storage """
    name = "SageNet50_CBAM_OTF_Feb03_dummy"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ RayTune (Untested) """
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
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = True
    )

    """ Epochs and Batches """
    num_epochs = 500
    batch_size = 64
    validation_plot_freq = 1 # every n epochs
    
    """ Weight Types """
    # Lowest loss, highest accuracy, highest auc, highest low_far_nsignals found
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
    freeze_for_transfer = False
    weights_path = 'weights_loss.pt'

    """ Optimizer """
    ## Adam 
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)

    """ Scheduler """
    ## Cosine Annealing with Warm Restarts
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    
    """ Gradient Clipping """
    clip_norm = 10000

    """ Automatic Mixed Precision """
    # Keep this turned off when using Adam
    # It seems to be unstable and produces NaN losses
    do_AMP = False

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True

    """ Loss Function """
    # All parameter estimation is done only using MSE loss at the moment
    loss_function = BCEWithPEregLoss(gw_loss=torch.nn.BCEWithLogitsLoss(), mse_alpha=1.0)
    
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
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    # Auxilliary noise data (only used for training, not for validation)
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )
    
    """ Transforms """
    batchshuffle_noise = False

    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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

    """ Optional things to do during training """
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

    # Debugging
    debug = False
    debug_size = 10000

    verbose = True



# DONE (BEST 24/06/24)
class SageNetOTF_May24_Russet(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "SageNet50_halfnormSNR_May17_Russet"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    # weights_path = "/home/nnarenraju/Research/ORChiD/RUNS/SageNet50_halfnormSNR_May17_Russet/checkpoint_epoch_39.pt"

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

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
    test_foreground_output = "testing_foutput_BEST_June.hdf"    
    test_background_output = "testing_boutput_BEST_June.hdf"


# DONE (Unbiased and bad noise rejection)
class SageNetOTF_metric_density_Desiree(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)
    # 3. Template placement density - fixed (**VARIATION**)

    """ Data storage """
    name = "SageNet50_metric_density_Jun26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

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
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:1'
    train_device = 'cuda:1'

    # Run device for testing phase
    testing_device = 'cuda:1'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_metric_latest.hdf"    
    test_background_output = "testing_boutput_metric_latest.hdf"


# Anneal from U(m1, m2) to template placement metric
class Kennebec_Annealed(SageNetOTF):
    # Hopefully we can keep both noise rejection capabilities and unbiased representation

    """ Data storage """
    name = "Kennebec_Annealed_training_Jul25"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    # Weights for testing
    weights_path = 'CHECKPOINTS/checkpoint_epoch_39.pt'

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:0',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:0'
    train_device = 'cuda:0'

    # Run device for testing phase
    testing_device = 'cuda:0'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_annealed_training.hdf"    
    test_background_output = "testing_boutput_annealed_training.hdf"


class Norland_D3_template_density(SageNetOTF):
    # Running D3 on template placement metric
    # Due to the abscence of blip glitches sensitivitiy should not suffer
    # PSD distribution should match exactly between train and test

    """ Data storage """
    name = "Norland_D3_template_placement_metric_Jul26"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

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
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    'validation': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    },
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:2',
        review = False
    )

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d3"
    test_foreground_output = "testing_foutput_D3_SageNet.hdf"    
    test_background_output = "testing_boutput_D3_SageNet.hdf"


class Russet_TrainingRecolour(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)
    # 3. Recoloured using training data (51 days)

    """ Data storage """
    name = "Russet_TrainingRecolour_Aug09"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_latter51days_20s.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_latter51days_20s.hdf"),
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
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = 'cuda:2',
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    
    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_training_recolour_Aug09.hdf"
    test_background_output = "testing_boutput_training_recolour_Aug09.hdf"


# ABLATION - fixed dataset (RUNNNG)
class Vitelotte_FixedDataset(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)
    # 3. Fixed dataset

    ## What is a fixed dataset?
    # 1. Sample seed does not change between epochs (DONE)
    # 2. Augmentation for signals is the same between epochs (DONE)
    # 3. No augmentation for noise (DONE)

    """ Data storage """
    name = "Vitelotte_FixedDataset_Aug31_smaller"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    batchshuffle_noise = True
    
    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = None,
        noise  = None,
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentPolSky(),
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = KappaModel_ResNet1D

    model_params = dict(
        # Resnet50
        resnet_size = 50,
        store_device = 'cuda:2',
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
    )
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Storage Devices """
    store_device = 'cuda:2'
    train_device = 'cuda:2'

    # Run device for testing phase
    testing_device = 'cuda:2'

    testing_dir = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
    test_foreground_output = "testing_foutput_fixed_dataset_smallnet_Sept12.hdf"
    test_background_output = "testing_boutput_fixed_dataset_smallnet_Sept12.hdf"


# ABLATION 1 - small network resnet50
class Butterball_ResNet1D(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)

    """ Data storage """
    name = "Butterball_ResNet1D_large_Sept30_withPE"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 64
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    seed_offset_train = 2**29
    seed_offset_valid = 2**26

    """ Generation """
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = KappaModel_ResNet1D

    model_params = dict(
        # Resnet50
        resnet_size = 152,
        store_device = torch.device('cuda:0'),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )

    """ Storage Devices """
    store_device = torch.device('cuda:0')
    train_device = torch.device('cuda:0')

    # Run device for testing phase
    testing_device = torch.device('cuda:0')
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_resnet1d_large_withPE_Sept30.hdf"    
    test_background_output = "testing_boutput_resnet1d_large_withPE_Sept30.hdf"


class Butterball_ResNet1D_withoutPE(SageNetOTF):
    ### Primary Deviations (Comparison to BOY latest) ###
    # 1. 113 days of O3b data (not variation)
    # 2. SNR halfnorm (not variation)

    """ Data storage """
    name = "Butterball_ResNet1D_Sept1_withoutPE"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    """ Generation """
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False,
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False,
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = KappaModel_ResNet1D

    model_params = dict(
        # Resnet50
        resnet_size = 50,
        store_device = torch.device('cuda:0'),
    )

    """ Storage Devices """
    store_device = torch.device('cuda:0')
    train_device = torch.device('cuda:0')

    # Run device for testing phase
    testing_device = torch.device('cuda:0')
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_resnet1d_withoutPE_Sept1.hdf"    
    test_background_output = "testing_boutput_resnet1d_withoutPE_Sept1.hdf"


class SageNetOTF_Aug27_Russet_diffseed_2(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "SageNet50_halfnormSNR_Sept11_Russet_diffseed_another"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')
    
    # testing on individual events from GWTC-3 confident
    batch_size = 1

    """ Dataset """ 
    dataset = MinimalOTF
    dataset_params = dict()

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    weights_path = 'weights_low_far_nsignals_39.pt'

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/H1",
                                                            L1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = torch.device("cuda:2"),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )

    """ Dataloader params """
    num_workers = 2
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True
    
    """ Storage Devices """
    store_device = torch.device("cuda:2")
    train_device = torch.device("cuda:2")

    # Run device for testing phase
    testing_device = torch.device("cuda:2")

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_BEST_June_diff_seed_Sept11_2.hdf"    
    test_background_output = "testing_boutput_BEST_June_diff_seed_Sept11_2.hdf"


class SageNetOTF_Feb05_Russet_nonastro(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "SageNet50_nonastro_Feb05_Russet"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    seed_offset_train = 2**25
    seed_offset_valid = 2**29

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/H1",
                                                            L1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = torch.device("cuda:0"),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )

    """ Dataloader params """
    num_workers = 32
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True
    
    """ Storage Devices """
    store_device = torch.device("cuda:0")
    train_device = torch.device("cuda:0")

    # Run device for testing phase
    testing_device = torch.device("cuda:0")

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_non_astro_best_retrain.hdf"    
    test_background_output = "testing_boutput_non_astro_best_retrain.hdf"


class SageNetOTF_Feb05_Russet_bayes_factor(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "SageNet50_bayes_factor_Feb06_Russet"
    export_dir = Path("/home/nnarenraju/Research/sgwc-1/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    seed_offset_train = 2**25
    seed_offset_valid = 2**29

    """ Loss Function """
    # Using lPOP loss for GW classification and MSE loss for point estimate of GW parameters
    loss_function = lPOPWithPEregLoss(lpop_alpha=2.0, lpop_beta=1.0, mse_alpha=1.0)

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/H1",
                                                            L1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = torch.device("cuda:1"),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )

    """ Dataloader params """
    num_workers = 32
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True
    
    """ Storage Devices """
    store_device = torch.device("cuda:1")
    train_device = torch.device("cuda:1")

    # Run device for testing phase
    testing_device = torch.device("cuda:1")

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_bayes_factor_best_retrain.hdf"    
    test_background_output = "testing_boutput_bayes_factor_best_retrain.hdf"


# Single epoch validation
class Validate_1epoch(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "TrainRecolour_BEST_1epoch_validation_Oct29"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    # Weights for testing
    pretrained = True
    weights_path = '/home/nnarenraju/Research/ORChiD/ML-GWSC1-Glasgow/WEIGHTS/weights_training_recolour_ep49.pt'
    seed_offset_train = 2**28
    seed_offset_valid = 2**28

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),

        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=0, segment_ulimit=132, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.0, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
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
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = torch.device("cuda:1"),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    

    """ Dataloader params """
    num_workers = 48
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    num_epochs = 1
    
    """ Storage Devices """
    store_device = torch.device("cuda:1")
    train_device = torch.device("cuda:1")

    # Run device for testing phase
    testing_device = torch.device("cuda:1")


# 1 signal runs for robustness to PSD
class Validate_1epoch_TrainRecolour(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "TrainRecolour_1epoch_validation_Sept7_traindata_longer"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    # Weights for testing
    pretrained = True
    weights_path = './WEIGHTS/weights_training_recolour_ep49.pt'

    """ Generation """
    signal_param = {'approximant': 'IMRPhenomPv2', 'f_ref': 20.0, 'mass1': 34.78719347340714, 'mass2': 26.213305061289596, 
                    'ra': 3.1059617471553547, 'dec': -0.6182337605590603, 'inclination': 0.8318006348684067, 
                    'coa_phase': 1.7465934740664464, 'polarization': 2.8848538945232653, 'chirp_distance': 240.66621748823462, 
                    'spin1_a': 0.3205292059008806, 'spin1_azimuthal': 2.791139350243793, 'spin1_polar': 1.5426111575393548, 
                    'spin2_a': 0.3809497458109482, 'spin2_azimuthal': 5.555278986183884, 'spin2_polar': 1.8106375482871242, 
                    'injection_time': 1242017656.1400197, 'tc': 11.085430996548288, 'spin1x': -0.3009269684153754, 
                    'spin1y': 0.11000153135040237, 'spin1z': 0.009032973837402562, 'spin2x': 0.2762643625638985, 
                    'spin2y': -0.24619412379548633, 'spin2z': -0.09049400101200968, 'mchirp': 26.236024494682457, 
                    'q': 1.3270815485521894, 'distance': 3106.193154419415}

    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         one_signal_params = signal_param,
                                         debug_me = False
                                        ),
                ]),

        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/local/scratch/igr/nnarenraju/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/local/scratch/igr/nnarenraju/O3b_real_noise/H1",
                                                            L1="/local/scratch/igr/nnarenraju/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.0, # 0.689 - 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0, fix_snr=10.0),
                ]),
        noise=UnifyNoise([
                    Recolour(use_precomputed=True, 
                             h1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_H1_30days.hdf"),
                             l1_psds_hdf=os.path.join(repo_abspath, "notebooks/tmp/psds_L1_30days.hdf"),
                             p_recolour=0.0, # 0.3829
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
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = torch.device("cuda:1"),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    num_epochs = 1
    
    """ Storage Devices """
    store_device = torch.device("cuda:1")
    train_device = torch.device("cuda:1")

    # Run device for testing phase
    testing_device = torch.device("cuda:1")


class Validate_1epoch_D3(SageNetOTF):
    ### Primary Deviations (Comparison to BOY) ###
    # 1. 113 days of O3b data (**VARIATION**)
    # 2. SNR halfnorm (**VARIATION**)

    """ Data storage """
    name = "Dataset3_1epoch_validation_Sept11_traindata"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    # Weights for testing
    pretrained = True
    weights_path = './WEIGHTS/weights_dataset3_ep49.pt'

    """ Generation """

    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),

        noise  = UnifyNoiseGen({
                    'training': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    'validation': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    },
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
                ]),
        noise=None,
        train=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=False, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        test=Unify({
                    'stage1':[
                            Whiten(trunc_method='hann', remove_corrupted=False, estimated=False),
                    ],
                    'stage2':[
                            Normalise(ignore_factors=True),
                            MultirateSampling(),
                    ],
                }),
        target=None
    )

    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = 'cuda:1',
        review = False
    )
    

    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    num_epochs = 1
    
    """ Storage Devices """
    store_device = torch.device("cuda:1")
    train_device = torch.device("cuda:1")

    # Run device for testing phase
    testing_device = torch.device("cuda:1")


class Validate_1epoch_D4_BEST_background_estimation_c1(SageNetOTF):

    """ Data storage """
    name = "Dataset4_1epoch_validation_Feb12_BEST_background_estimation_chunk_1"
    export_dir = Path("/home/nnarenraju/Research/sgwc-1/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    # Save weights for particular epochs
    save_epoch_weight = list(range(4, 100, 5))

    seed_offset_train = 2**25
    seed_offset_valid = 2**29

    # Weights for testing
    pretrained = True
    weights_path = "/home/nnarenraju/Research/ORChiD/ML-GWSC1-Glasgow/WEIGHTS/weights_BEST_diffseed_Sept1_39.pt"

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    'validation': RandomNoiseSlice(
                                    real_noise_path="/home/nnarenraju/Research/ORChiD/O3a_real_noise/O3a_real_noise.hdf",
                                    segment_llimit=133, segment_ulimit=-1, debug_me=False
                                ),
                    },
                    MultipleFileRandomNoiseSlice(noise_dirs=dict(
                                                            H1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/H1",
                                                            L1="/home/nnarenraju/Research/ORChiD/O3b_real_noise/L1",
                                                        ),
                                                 debug_me=False,
                                                 debug_dir=""
                    ),
                    paux = 0.689, # 113/164 days for extra O3b noise
                    debug_me=False,
                    debug_dir=os.path.join(debug_dir, 'NoiseGen')
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy_minimal

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = torch.device("cuda:0"),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    

    """ Dataloader params """
    num_workers = 32
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True

    num_epochs = 1
    batch_size = 16
    
    """ Storage Devices """
    store_device = torch.device("cuda:0")
    train_device = torch.device("cuda:0")

    # Run device for testing phase
    testing_device = torch.device("cuda:0")


class Vitelotte_FixedDataset_Aug24(SageNetOTF):

    """ Data storage """
    name = "Vitelotte_FixedDataset_Aug24"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    batchshuffle_noise = True
    
    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = None,
        noise  = None,
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentPolSky(),
                    AugmentOptimalNetworkSNR(rescale=True, use_uniform=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM_legacy

    model_params = dict(
        # Resnet50
        filter_size = 32,
        kernel_size = 64,
        resnet_size = 50,
        store_device = torch.device("cuda:1"),
        parameter_estimation = ('norm_tc', 'norm_mchirp', )
    )
    
    """ Dataloader params """
    num_workers = 16
    pin_memory = True
    prefetch_factor = 8
    persistent_workers = True

    """ Storage Devices """
    store_device = torch.device("cuda:1")
    train_device = torch.device("cuda:1")

    # Run device for testing phase
    testing_device = torch.device("cuda:1")

    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d4"
    test_foreground_output = "testing_foutput_Vitelotte_FixedDataset_Aug24.hdf"
    test_background_output = "testing_boutput_Vitelotte_FixedDataset_Aug24.hdf"


class Norland_D3_BEST_settings(SageNetOTF):
    # Running D3 on template placement metric
    # Due to the abscence of blip glitches sensitivitiy should not suffer
    # PSD distribution should match exactly between train and test

    """ Data storage """
    name = "Norland_D3_BEST_settings_Sept11"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

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
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    'validation': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/psds")),
                    },
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = torch.device('cuda'),
        review = False
    )

    """ Storage Devices """
    store_device = torch.device('cuda')
    train_device = torch.device('cuda')

    # Run device for testing phase
    testing_device = torch.device('cuda')
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d3"
    test_foreground_output = "testing_foutput_D3_SageNet_BEST_settings.hdf"    
    test_background_output = "testing_boutput_D3_SageNet_BEST_settings.hdf"


class Norland_D3_Odds_Ratio(SageNetOTF):
    # Running D3 on template placement metric
    # Due to the abscence of blip glitches sensitivitiy should not suffer
    # PSD distribution should match exactly between train and test

    """ Data storage """
    name = "Norland_D3_Odds_Ratio_Feb18"
    export_dir = Path("/home/nnarenraju/Research/ORChiD/RUNS") / name
    debug_dir = "./DEBUG"
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = os.path.join(git_revparse.stdout.strip('\n'), 'sage')

    """ Dataset """
    dataset = MinimalOTF
    dataset_params = dict()

    """ Dataloader params """
    num_workers = 32
    pin_memory = True
    prefetch_factor = 4
    persistent_workers = True

    """ Generation """
    # Augmentation using GWSPY glitches happens only during training (not for validation)
    generation = dict(
        signal = UnifySignalGen([
                    FastGenerateWaveform(rwrap = 3.0, 
                                         beta_taper = 8, 
                                         pad_duration_estimate = 1.1, 
                                         min_mass = 5.0, 
                                         debug_me = False
                                        ),
                ]),
        noise  = UnifyNoiseGen({
                    'training': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/limited_psds")),
                    'validation': ColouredNoiseGenerator(psds_dir=os.path.join(repo_abspath, "data/limited_psds")),
                    },
                )
    )

    """ Transforms """
    transforms = dict(
        signal=UnifySignal([
                    AugmentOptimalNetworkSNR(rescale=True, use_halfnorm=True, snr_lower_limit=5.0, snr_upper_limit=15.0),
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
    
    """ Architecture """
    model = Rigatoni_MS_ResNetCBAM

    # Following options available for pe point estimate
    # 'norm_tc', 'norm_dchirp', 'norm_mchirp', 
    # 'norm_dist', 'norm_q', 'norm_invq', 'norm_snr'
    model_params = dict(
        scales = [1, 2, 4, 0.5, 0.25],
        blocks = [
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock], 
            [MultiScaleBlock, MultiScaleBlock]
        ],
        out_channels = [[32, 32], [64, 64], [128, 128]],
        base_kernel_sizes = [
            [64, 64 // 2 + 1], 
            [64 // 2 + 1, 64 // 4 + 1], 
            [64 // 4 + 1, 64 // 4 + 1]
        ], 
        compression_factor = [8, 4, 0],
        in_channels = 1,
        resnet_size = 50,
        parameter_estimation = ('norm_tc', 'norm_mchirp', ),
        norm_layer = 'instancenorm',
        store_device = torch.device('cuda:0'),
        review = False
    )

    """ Storage Devices """
    store_device = torch.device('cuda:0')
    train_device = torch.device('cuda:0')

    # Run device for testing phase
    testing_device = torch.device('cuda:0')
    
    testing_dir = "/home/nnarenraju/Research/ORChiD/test_data_d3"
    test_foreground_output = "testing_foutput_D3_SageNet_odds_ratio.hdf"    
    test_background_output = "testing_boutput_D3_SageNet_odds_ratio.hdf"
