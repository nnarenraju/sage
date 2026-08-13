#!/usr/bin/env python
"""Decisive disambiguation: compare each detector's raw/whitened peak to its
TRUE expected arrival tc+dt (not geocentric tc). Removes the projection-delay
confound in the sum-over-detectors metric.
"""
import os, sys, json, math
import numpy as np
import torch

torch.manual_seed(0); np.random.seed(0)
sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")

import torch as _t
from sage.core.config import register_configs, get_cfg, get_data_cfg
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()

class O3bCFG:
    export_dir = "/home/nagarajan/research/sage/sage/diagnostics/plots"
    fiducial_dir = "/work/nagarajan/sage_runs/fiducial_psds_o3ab"
    batch_size = 128; device = "cpu"; dtype = _t.float32
    detectors = ["H1", "L1"]; train_runs = ["O3b"]
    do_point_estimate = ["tc", "mchirp"]; autocast = False
    class_balance = 0.5; clip_norm = 1.0; dropout = 0.0; num_epochs = 10
    training_iterations = int(2_000_000 / batch_size)
    validation_iterations = int(200_000 / batch_size)

class O3bDataCFG:
    data_dir = _SRV.data_dir("O3b")
    training_noise_files = [_SRV.noise_bin(d, "O3b") for d in O3bCFG.detectors]
    validation_noise_files = [_SRV.noise_bin(d, "O3a", "data_release_o3a") for d in O3bCFG.detectors]
    sample_rate = 2048.0; noise_low_frequency_cutoff = 15.0
    signal_low_frequency_cutoff = 20.0
    sample_length_in_s = 12.0; padding_length_in_s = 2.0

register_configs(BaseConfig(O3bCFG()), BaseDataConfig(O3bDataCFG()))
cfg, data_cfg = get_cfg(), get_data_cfg()
SR = data_cfg.sample_rate

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2, HalfNorm
from sage.data.waveform.snr import OptimalSNRRescaler
from sage.dsp.whiten import FiducialWhitening

param_sampler = read_from_config("./gwconfig.yaml", seed=150914)
snrscaler = OptimalSNRRescaler(HalfNorm(scale=4.0, loc=5.0, seed=150914))
# append_per_det_targets=True so forward returns dt via targets layout
signal_sampler = IMRPhenomPv2(param_sampler, ConstantProjection(), augment=snrscaler,
                              append_per_det_targets=True)
tc_col = signal_sampler.tc_col
mchirp_col = param_sampler.param_index["mchirp"]
whitener = FiducialWhitening()

def fd_to_td_raw(hf):
    x_td = torch.fft.irfft(hf, dim=-1, norm="forward") * whitener.delta_f
    return whitener.remove_corrupted(x_td)

def perdet_peak_time(td_np):  # (N,D,VALID) -> (N,D) seconds
    return np.abs(td_np).argmax(axis=-1) / SR

S = signal_sampler.B
N_TARGET = 384
nb = math.ceil(N_TARGET / S)
tc_l, mc_l, dt_l = [], [], []
raw_pk_l, wh_pk_l = [], []
for i in range(nb):
    all_theta = signal_sampler.param_sampler(S)
    req_theta = all_theta[:, signal_sampler.req_idx]
    hp, hc = signal_sampler.get_hphc(req_theta)
    hf, dt = signal_sampler.waveform_project(
        hp, hc, ra=req_theta[:, -2], dec=req_theta[:, -1],
        polarization=req_theta[:, -3], return_delay=True)
    hf, scale = signal_sampler.augment(hf)
    tc_l.append(all_theta[:, tc_col].cpu().numpy())
    mc_l.append(all_theta[:, mchirp_col].cpu().numpy())
    dt_l.append(dt.cpu().numpy())               # (S,D)
    wtd = whitener(hf).cpu().numpy()
    rtd = fd_to_td_raw(hf).cpu().numpy()
    wh_pk_l.append(perdet_peak_time(wtd))
    raw_pk_l.append(perdet_peak_time(rtd))

tc = np.concatenate(tc_l)[:N_TARGET]
mc = np.concatenate(mc_l)[:N_TARGET]
dt = np.concatenate(dt_l)[:N_TARGET]            # (N,D)
raw_pk = np.concatenate(raw_pk_l)[:N_TARGET]    # (N,D)
wh_pk = np.concatenate(wh_pk_l)[:N_TARGET]
N, D = raw_pk.shape
tc_exp = tc[:, None] + dt                        # (N,D) true per-detector arrival

def stats(a):
    a = a.ravel()
    return dict(median=float(np.median(a)), std=float(np.std(a)),
                absmax=float(np.abs(a).max()), mean=float(np.mean(a)),
                frac_within_5ms=float((np.abs(a) < 5).mean()),
                frac_within_20ms=float((np.abs(a) < 20).mean()))

# residual vs TRUE per-detector arrival (removes dt confound)
raw_res = (raw_pk - tc_exp) * 1e3
wh_res = (wh_pk - tc_exp) * 1e3
# naive vs geocentric tc (with dt confound) for reference
raw_naive = (raw_pk - tc[:, None]) * 1e3
wh_naive = (wh_pk - tc[:, None]) * 1e3

print("dt range [ms]:", float(dt.min()*1e3), float(dt.max()*1e3), "median", float(np.median(dt)*1e3))
print("RAW  peak - (tc+dt) [ms]:", json.dumps(stats(raw_res), indent=2))
print("WHIT peak - (tc+dt) [ms]:", json.dumps(stats(wh_res), indent=2))
print("RAW  peak - tc(geo) [ms]:", json.dumps(stats(raw_naive), indent=2))
print("WHIT peak - tc(geo) [ms]:", json.dumps(stats(wh_naive), indent=2))

# mass correlation of the TRUE residual (per-detector flattened)
mc_flat = np.repeat(mc, D)
def corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])
print("corr(mc, raw_res)=", corr(mc_flat, raw_res.ravel()),
      " corr(mc, wh_res)=", corr(mc_flat, wh_res.ravel()))

out = {"N": N, "D": D,
       "raw_peak_minus_tc_plus_dt_ms": stats(raw_res),
       "whitened_peak_minus_tc_plus_dt_ms": stats(wh_res),
       "raw_peak_minus_geocentric_tc_ms": stats(raw_naive),
       "whitened_peak_minus_geocentric_tc_ms": stats(wh_naive),
       "dt_median_ms": float(np.median(dt)*1e3),
       "corr_mc_raw_res": corr(mc_flat, raw_res.ravel()),
       "corr_mc_wh_res": corr(mc_flat, wh_res.ravel())}
with open("/home/nagarajan/research/sage/sage/diagnostics/plots/perdet_tc.json", "w") as f:
    json.dump(out, f, indent=2)
print("=== DONE ===")
