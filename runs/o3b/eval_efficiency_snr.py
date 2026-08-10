#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_efficiency_snr.py -- thin driver for a SageVanillaTesting sensitivity run.

Loads a captured checkpoint and runs SageVanillaTesting (large-batch NOISE +
SIGNAL passes). Confound-free by construction: signals are rescaled to a broad
UNIFORM SNR using the COMMON production fiducial (both runs share one SNR axis),
while the model whitens with its OWN fiducial (as trained). Writes
``testing_data.h5`` to a fresh eval dir -- never the live run's export_dir.

Env:
  SAGE_CONFIG      run config module (config_HL | config_HL_yearold)
  EVAL_CKPT        path to epoch_NN.pt
  EVAL_OUT         fresh output dir
  EVAL_COMMON_FID  common fiducial dir for the SNR axis (default: prod o3ab)
  EVAL_PASS        noise | signal | both (default both)
  N_NOISE          noise windows   (default 1_000_000)
  N_SIG            signal windows  (default 100_000)
  EVAL_BATCH       inference batch (default 1024)

__license__ = GPL-3.0-or-later
"""
import os
import importlib

import numpy as np
import torch

importlib.import_module(os.environ["SAGE_CONFIG"]).set_configs()
from sage.core.config import get_cfg, get_data_cfg

cfg, dcfg = get_cfg(), get_data_cfg()

EVAL_CKPT  = os.environ["EVAL_CKPT"]
EVAL_OUT   = os.environ["EVAL_OUT"]; os.makedirs(EVAL_OUT, exist_ok=True)
COMMON_FID = os.environ.get("EVAL_COMMON_FID",
                            "/work/nagarajan/sage_runs/fiducial_psds_o3ab")
N_NOISE    = int(float(os.environ.get("N_NOISE", "1000000")))
N_SIG      = int(float(os.environ.get("N_SIG", "100000")))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "1024"))
SNR_LO, SNR_HI = 3.0, 25.0
SEED = 424242

# Redirect output + enlarge the inference batch, and make the signal sampler emit
# a full batch (class_balance=1.0) so signal-in-noise batches align with the noise
# batch. BaseConfig has no __setattr__, so these shadow the delegate; never touch
# the live run's files.
cfg.export_dir = EVAL_OUT
cfg.batch_size = EVAL_BATCH
cfg.class_balance = 1.0

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.data.noise import TestNoiseSampler
from sage.core.graph import Preprocessor
from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning
from sage.architecture.network import MSCNN1D_2DResNetCBAM_HardMining
from sage.factory import SageVanillaTesting

device = cfg.device


class UniformSNR:
    """Broad uniform target-SNR sampler that records its last draw."""
    def __init__(self, lo, hi, seed):
        self.lo, self.hi = float(lo), float(hi)
        self.g = torch.Generator().manual_seed(int(seed))
        self.last = None

    def __call__(self, n):
        v = self.lo + (self.hi - self.lo) * torch.rand(n, generator=self.g)
        self.last = v
        return v


def load_asds(fid_dir):
    arrs = [np.fromfile(os.path.join(fid_dir, f"fiducial_{d}_psd.bin"),
                        dtype=np.float32) for d in cfg.detectors]
    return torch.from_numpy(np.stack(arrs, 0)).to(device).unsqueeze(0)  # (1,D,F)


# ---- model + checkpoint (strip torch.compile "_orig_mod." if present) --------
model = MSCNN1D_2DResNetCBAM_HardMining(
    frontend_filters=32, frontend_kernel=64, backend_resnet_size=50,
    norm_type=cfg.norm_type, dropout=cfg.dropout,
).to(dtype=cfg.dtype, device=device, memory_format=torch.channels_last)
ck = torch.load(EVAL_CKPT, map_location=device, weights_only=False)
sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}
model.load_state_dict(sd)
print(f"[{os.environ['SAGE_CONFIG']}] ckpt={EVAL_CKPT} epoch={ck.get('epoch')} "
      f"norm={cfg.norm_type} batch={EVAL_BATCH} dets={cfg.detectors} "
      f"N_noise={N_NOISE:,} N_sig={N_SIG:,}", flush=True)

# ---- signal path: uniform SNR via the COMMON fiducial ------------------------
param_sampler = read_from_config("./gwconfig.yaml", seed=SEED)
snrscaler = OptimalSNRRescaler(UniformSNR(SNR_LO, SNR_HI, SEED))
snrscaler.snr_estimator.asds = load_asds(COMMON_FID)          # COMMON SNR axis
signal_sampler = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler)

# ---- noise + processor (whitener uses the RUN'S OWN fiducial) ----------------
noise_sampler = TestNoiseSampler(postprocess_fn=None, prefetch=8,
                                 seed=SEED + 1, training=False)
processor = Preprocessor([FiducialWhitening(),
                          MultirateSampler(binning_method=DyadicPyramidBinning(
                              param_sampler.bounds))])

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

KEEP_LOUD = int(float(os.environ.get("EVAL_KEEP_LOUD", "5000")))
tester = SageVanillaTesting(signal_sampler, noise_sampler, processor, model,
                            amp_dtype=amp_dtype, keep_loud=KEEP_LOUD)

# EVAL_PASS = noise | signal | both -- run each pass independently at its own count.
PASS = os.environ.get("EVAL_PASS", "both").lower()
npath = os.path.join(EVAL_OUT, "testing_noise.h5")
if PASS in ("noise", "both"):
    tester.run_noise(N_NOISE, save_path=npath)
if PASS in ("signal", "both"):
    tester.run_signal(N_SIG, save_path=os.path.join(EVAL_OUT, "testing_signal.h5"))

# ---- reconstruct self-check: metadata -> exact strain -> re-score == recorded ----
if PASS in ("noise", "both"):
    try:
        import h5py
        from sage.factory import reconstruct_noise
        nchk = 5
        strain = reconstruct_noise(npath, noise_sampler, indices=np.arange(nchk))  # (nchk,D,L) physical
        with h5py.File(npath) as f:
            recorded = f["loud_scores"][:nchk]
        with torch.inference_mode():
            xfd = torch.fft.rfft(torch.from_numpy(strain).to(device), dim=-1, norm="forward")
            rescored = tester._forward(xfd)[:, 0].detach().cpu().numpy()
        print(f"[reconstruct check] recorded={np.round(recorded,4)} "
              f"rescored={np.round(rescored,4)} max|d|={float(np.max(np.abs(recorded-rescored))):.4g}",
              flush=True)
    except Exception as e:
        print(f"[reconstruct check] SKIPPED: {type(e).__name__}: {e}", flush=True)

print(f"done ({PASS}) -> {EVAL_OUT}", flush=True)
