#!/usr/bin/env python
"""Verify Sage's optimal-SNR estimator against PyCBC `sigmasq` (the reference).

Confirms that the PADDED delta_f (1/16 = 1/padded_length) used by
OptimalSNREstimator is the physically-correct frequency resolution of the padded
FD waveform: Sage(padded) == PyCBC sigmasq to machine precision, while the old
unpadded delta_f (1/12 = 1/sample_length) under-reports optimal SNR by ~13.4%
(=sqrt(12/16)) and integrates the wrong band. Run on CPU. Prints per-case
numbers + saves plots/snr_vs_pycbc.png.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "/home/nagarajan/research/sage")
os.chdir("/home/nagarajan/research/sage/runs/o3b")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from sage.core.config import register_configs
from sage.core.base_classes import BaseConfig, BaseDataConfig
from sage.utils.servers import get_server
_SRV = get_server()
OUT = "/home/nagarajan/research/sage/sage/diagnostics/plots"; os.makedirs(OUT, exist_ok=True)

class C:
    export_dir=OUT; fiducial_dir="/home/nagarajan/research/sage/runs/o3b/run_export/fiducial_psds"
    batch_size=8; device="cpu"; dtype=torch.float32; detectors=["H1","L1"]; train_runs=["O3b"]
    do_point_estimate=["tc","mchirp"]; autocast=False; class_balance=0.5; clip_norm=1.0; dropout=0.0
    num_epochs=1; training_iterations=10; validation_iterations=1
class DC:
    data_dir=_SRV.data_dir("O3b"); training_noise_files=[_SRV.noise_bin(d,"O3b") for d in C.detectors]
    validation_noise_files=[_SRV.noise_bin(d,"O3a","data_release_o3a") for d in C.detectors]
    sample_rate=2048.0; noise_low_frequency_cutoff=15.0; signal_low_frequency_cutoff=20.0
    sample_length_in_s=12.0; padding_length_in_s=2.0
register_configs(BaseConfig(C()), BaseDataConfig(DC()))

from sage.data.waveform import read_from_config, ConstantProjection, IMRPhenomPv2
from sage.data.waveform.snr import OptimalSNREstimator
from sage.data.psd import get_fiducial_psds

DF_PAD = 1.0/16.0       # padded_delta_f = 1/(sample_length + 2*padding) -- CORRECT
DF_UNPAD = 1.0/12.0     # old unpadded delta_f = 1/sample_length -- WRONG
F_LOW, F_HIGH = 20.0, 1024.0

# --- generate a few raw projected waveforms (no SNR rescaling) ---
ps = read_from_config("./gwconfig.yaml", seed=150914)
ss = IMRPhenomPv2(ps, ConstantProjection(), augment=None)
hf, _ = ss()                                   # (S, D, F) padded grid, 16385 bins
hf = hf.detach().cpu().numpy()
asds = get_fiducial_psds().detach().cpu().numpy()   # (D, F) = sqrt(PSD)
S, D, F = hf.shape

def sage_snr(h1d, asd1d, df):
    """Sage OptimalSNREstimator formula with an arbitrary delta_f + its mask."""
    k_lo = int(np.ceil(F_LOW/df)); k_hi = int(np.floor(F_HIGH/df))
    hw = (h1d/df)/asd1d
    p = (hw.real**2 + hw.imag**2)
    m = np.zeros(F); m[k_lo:k_hi] = 1.0
    return float(np.sqrt(4.0*df*np.sum(p*m)))

def pycbc_snr(h1d, asd1d, df):
    """PyCBC sigmasq reference. h_stored = h_continuousFT * df, so htilde = h/df.
    NOTE: square the ASD in float64 -- ASD~6e-24 squared underflows float32 to 0."""
    from pycbc.types import FrequencySeries
    from pycbc.filter import sigmasq
    htilde = FrequencySeries((h1d.astype(np.complex128)/df), delta_f=df)
    psd = FrequencySeries(asd1d.astype(np.float64)**2, delta_f=df)   # square in fp64
    return float(np.sqrt(sigmasq(htilde, psd=psd, low_frequency_cutoff=F_LOW,
                                 high_frequency_cutoff=F_HIGH)))

# production estimator (network rho over both detectors; sanity that it runs)
est = OptimalSNREstimator(); rho_prod, rho_prod_det = est(torch.from_numpy(hf))
rho_prod_det = rho_prod_det.numpy()   # (S, D) per-detector, uses padded 1/16

print(f"{'case':>8} | {'padded(1/16)':>12} {'unpad(1/12)':>11} {'pycbc':>9} | "
      f"{'pad/pycbc':>9} {'unpad/pycbc':>11}")
rows=[]
for s in range(S):
    h = hf[s,0]; a = asds[0]                      # detector 0 (H1)
    rp = sage_snr(h,a,DF_PAD); ru = sage_snr(h,a,DF_UNPAD); rc = pycbc_snr(h,a,DF_PAD)
    print(f"{s:8d} | {rp:12.5f} {ru:11.5f} {rc:9.5f} | {rp/rc:9.6f} {ru/rc:11.6f}")
    rows.append((rp,ru,rc))
rows=np.array(rows)
pad_ratio = float(np.median(rows[:,0]/rows[:,2])); unpad_ratio = float(np.median(rows[:,1]/rows[:,2]))
print(f"\nMEDIAN padded/pycbc  = {pad_ratio:.6f}   (1.000000 => padded is CORRECT)")
print(f"MEDIAN unpadded/pycbc= {unpad_ratio:.6f}   (~0.866 = sqrt(12/16) => under-reports ~13.4%)")
# cross-check: production estimator's per-detector H1 SNR == manual padded H1
manual_pad_h1 = np.array([sage_snr(hf[s,0], asds[0], DF_PAD) for s in range(S)])
print(f"production estimator (H1 per-det) vs manual padded H1: max|diff| = "
      f"{np.max(np.abs(rho_prod_det[:,0]-manual_pad_h1)):.2e}")

fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(rows[:,2], rows[:,0], "o", label="Sage padded 1/16");
ax[0].plot(rows[:,2], rows[:,1], "s", label="Sage unpadded 1/12")
lim=[0, rows[:,2].max()*1.1]; ax[0].plot(lim,lim,"k--",lw=0.8); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("PyCBC sigmasq SNR"); ax[0].set_ylabel("Sage SNR"); ax[0].legend(); ax[0].set_title("Sage vs PyCBC optimal SNR")
ax[1].bar(["padded\n1/16","unpadded\n1/12"], [pad_ratio, unpad_ratio], color=["C0","C3"])
ax[1].axhline(1.0, color="k", ls="--", lw=0.8); ax[1].set_ylabel("ratio to PyCBC"); ax[1].set_title("SNR ratio vs reference")
plt.tight_layout(); plt.savefig(f"{OUT}/snr_vs_pycbc.png", dpi=120); plt.close()
json.dump({"padded_over_pycbc_median":pad_ratio,"unpadded_over_pycbc_median":unpad_ratio,
           "n_cases":int(S)}, open(f"{OUT}/snr_vs_pycbc.json","w"), indent=2)
print(f"\nsaved {OUT}/snr_vs_pycbc.png")
print("DONE")
