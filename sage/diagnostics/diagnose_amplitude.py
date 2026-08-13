#!/usr/bin/env python
"""Decisive scale-consistency check: FiducialWhitening TD scale vs OptimalSNREstimator."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")

from sage.core.config import register_configs, get_cfg, get_data_cfg
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()
class C:
    export_dir='/home/nagarajan/research/sage/sage/diagnostics/plots'; fiducial_dir='/work/nagarajan/sage_runs/fiducial_psds_o3ab'
    batch_size=128; device='cpu'; dtype=torch.float32; detectors=['H1','L1']; train_runs=['O3b']
    do_point_estimate=['tc','mchirp']; autocast=False; class_balance=0.5; clip_norm=1.0; dropout=0.0
    num_epochs=10; training_iterations=100; validation_iterations=10
class DC:
    data_dir=_SRV.data_dir('O3b'); training_noise_files=[_SRV.noise_bin(d,'O3b') for d in C.detectors]
    validation_noise_files=[_SRV.noise_bin(d,'O3a','data_release_o3a') for d in C.detectors]
    sample_rate=2048.0; noise_low_frequency_cutoff=15.0; signal_low_frequency_cutoff=20.0
    sample_length_in_s=12.0; padding_length_in_s=2.0
register_configs(BaseConfig(C()),BaseDataConfig(DC()))

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler, OptimalSNREstimator
from sage.data.noise import MemmapNoiseSampler
from sage.dsp.whiten import FiducialWhitening

ps = read_from_config('./gwconfig.yaml', seed=42)
ss = IMRPhenomPv2(ps, ConstantProjection(), augment=OptimalSNRRescaler(HalfNorm(4.0,5.0,seed=42)))
wh = FiducialWhitening()
est = OptimalSNREstimator()

# --- signal scale consistency: optimal SNR (FD) vs TD whitened energy ---
hf, tg, th = ss(return_theta=True)
rho_net, rho_det = est(hf)                       # (S,), (S,D)
ws = wh(hf).detach().cpu().numpy()               # (S,D,VALID) whitened TD signal
# per-detector TD whitened "SNR" = sqrt(sum sw^2)
td_snr_det = np.sqrt((ws**2).sum(axis=2))        # (S,D)
td_snr_net = np.sqrt((td_snr_det**2).sum(axis=1))
rho_net_np = rho_net.detach().cpu().numpy()
ratio = td_snr_net / rho_net_np
print("=== SIGNAL SCALE CONSISTENCY ===")
print(f"optimal SNR (FD estimator):   mean={rho_net_np.mean():.3f} median={np.median(rho_net_np):.3f}")
print(f"TD whitened energy sqrt:       mean={td_snr_net.mean():.3f} median={np.median(td_snr_net):.3f}")
print(f"ratio TD/FD:                   mean={ratio.mean():.4f} median={np.median(ratio):.4f} std={ratio.std():.4f}")
print("  (ratio==1 -> FiducialWhitening TD scale is consistent with the SNR estimator;")
print("   ratio!=1 -> whitened data fed to the net is at a different scale than the SNR implies)")

# --- whitened noise std distribution (plain O3b noise) ---
ns = MemmapNoiseSampler(postprocess_fn=None, prefetch=4, seed=42)
stds = []
for _ in range(12):
    fd, _ = ns()
    wtd = wh(fd).detach().cpu().numpy()
    stds.append(wtd.std(axis=2).ravel())
ns.shutdown()
stds = np.concatenate(stds)
print("\n=== WHITENED NOISE STD (plain O3b, per sample-detector) ===")
for p in (1,5,10,25,50,75,90,95,99):
    print(f"  p{p:02d} = {np.percentile(stds,p):.3f}")
print(f"  mean={stds.mean():.3f} (heavy glitch tail); robust median={np.median(stds):.3f}")
# robust: fraction of near-unit segments
print(f"  frac in [0.8,1.2]={float(((stds>0.8)&(stds<1.2)).mean()):.3f}")
print(f"  frac < 0.6      ={float((stds<0.6).mean()):.3f}")
print(f"  frac > 2.0      ={float((stds>2.0).mean()):.3f}")

out = {
  "signal_scale": {"optimal_snr_median": float(np.median(rho_net_np)),
                   "td_whitened_energy_sqrt_median": float(np.median(td_snr_net)),
                   "ratio_td_over_fd_median": float(np.median(ratio)),
                   "ratio_td_over_fd_mean": float(ratio.mean())},
  "whitened_noise_std_pcts": {str(p): float(np.percentile(stds,p)) for p in (5,25,50,75,95)},
  "whitened_noise_std_mean": float(stds.mean()),
  "whitened_noise_frac_near_unit_0.8_1.2": float(((stds>0.8)&(stds<1.2)).mean()),
}
json.dump(out, open("/home/nagarajan/research/sage/sage/diagnostics/plots/amp_check.json","w"), indent=2)
print("\nJSON:", json.dumps(out, indent=2))
print("DONE")
