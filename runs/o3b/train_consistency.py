#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
o3b run script for the multi-detector consistency model.

Same data/preprocessing as ``train.py`` but:
  - the signal sampler is built with ``append_per_det_tc=True`` so the targets
    carry per-detector arrival times,
  - the model is :class:`MSCNN1D_2DResNetCBAM_Consistency` (fed the multirate
    ``t_grid``),
  - training uses :class:`SageConsistencyTraining` = BCE + merged-PE loss
    (``BCEWithPEsigmaLoss``) plus the separate per-detector ``ConsistencyNLLLoss``.

Launch with ``python3 -c "from train_consistency import run_consistency_sage; run_consistency_sage()"``.
"""

import os
import torch

torch.backends.cudnn.benchmark = True

from sage.core.config import get_cfg, get_data_cfg

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.data.noise import MemmapNoiseSampler, RecolourPostprocess

from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning
from sage.core.graph import Preprocessor

from sage.architecture.network import MSCNN1D_2DResNetCBAM_Consistency
from sage.architecture.custom_losses import BCEWithPEsigmaLoss, ConsistencyNLLLoss
from sage.core.logger import HDF5LossLogger

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sage.factory import SageConsistencyTraining

from config import set_configs
from sage.utils.checkpoint import CheckpointManager

_RECOLOUR_DIR = "/local/scratch/igr/nnarenraju/data_release/o3a_dataset"


def make_training_graph():
    # Signal sampler — append_per_det_tc=True so targets carry per-detector tc.
    param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
    snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
    signal_sampler = IMRPhenomPv2(
        param_sampler,
        ConstantProjection(),
        augment=snrscaler,
        append_per_det_tc=True,
    )

    recolour = RecolourPostprocess(p_recolour=0.37, recolour_dataset_dir=_RECOLOUR_DIR)
    noise_sampler = MemmapNoiseSampler(postprocess_fn=recolour, prefetch=8, seed=150914)

    return signal_sampler, noise_sampler, param_sampler.bounds


def make_processor(bounds):
    # Build the multirate stage once; reuse its time grid for the heads so the
    # tc soft-argmax t_position matches the actual net_input.
    whitener = FiducialWhitening()
    dyadic_binning = DyadicPyramidBinning(bounds)
    mrsampler = MultirateSampler(binning_method=dyadic_binning)
    processor = Preprocessor([whitener, mrsampler])
    return processor, mrsampler.output_time_grid()


def run_consistency_sage():

    set_configs()
    cfg, data_cfg = get_cfg(), get_data_cfg()

    signal_sampler, noise_sampler, bounds = make_training_graph()
    processor, t_grid = make_processor(bounds)

    model = MSCNN1D_2DResNetCBAM_Consistency(
        t_grid,
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="groupnorm",
        dropout=cfg.dropout,
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    print("Model compiled with torch.compile!")

    merged_loss = BCEWithPEsigmaLoss(regression_weight=0.005, coupling_weight=0.005)
    consistency_loss = ConsistencyNLLLoss(tc_weight=1.0, mc_weight=1.0)

    optimiser = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6, fused=True)
    scheduler = CosineAnnealingWarmRestarts(optimiser, T_0=5, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

    train_sage = SageConsistencyTraining(
        signal_sampler,
        noise_sampler,
        processor,
        model,
        merged_loss,
        consistency_loss,
        optimiser,
        scheduler,
        scaler,
        num_iterations=cfg.training_iterations,
        num_epochs=cfg.num_epochs,
        consistency_weight=0.1,
    )

    ckpt_mgr = CheckpointManager(
        cfg=cfg, data_cfg=data_cfg, model=model,
        optimizer=optimiser, scheduler=scheduler, scaler=scaler,
    )

    logger = HDF5LossLogger(
        path=os.path.join(cfg.export_dir, "losses.h5"),
        num_epochs=cfg.num_epochs,
        num_components=train_sage.num_components,
    )

    for nepoch in range(cfg.num_epochs):
        print(f"Epoch {nepoch}: Consistency training")
        train_sage(nepoch=nepoch)
        logger.log(train_sage.loss_components, nepoch, split="training")
        ckpt_mgr.save(
            epoch=nepoch,
            val_loss=float(train_sage.loss_components[nepoch][0].item()),
        )
