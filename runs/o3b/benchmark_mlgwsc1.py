#!/usr/bin/env python3
"""
MLGWSC-1 benchmark runner for the O3b model (epoch 0079, lowest validation loss).

Run from the runs/o3b directory:
    python3 benchmark_mlgwsc1.py

Results are saved to run_export/benchmark-mlgwsc1/.
"""

import os
import sys
import h5py
import torch
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext

import pycbc.types

RUN_DIR = os.path.dirname(os.path.abspath(__file__))
SAGE_DIR = os.path.join(RUN_DIR, "..", "..")
sys.path.insert(0, RUN_DIR)
sys.path.insert(0, SAGE_DIR)

# --- Config setup (must happen before any sage imports that use configs) ---
import os as _os, importlib as _il
# Config module to run: named by SAGE_CONFIG (default "config"). Make a
# per-network config with `cp config.py config_<DETS>.py`, edit `detectors`
# + `export_dir`, and pick it at launch. No network logic lives in code.
set_configs = _il.import_module(_os.environ.get("SAGE_CONFIG", "config")).set_configs
set_configs()

from sage.core.config import get_cfg, get_data_cfg

cfg = get_cfg()
data_cfg = get_data_cfg()

# Extend data_cfg with benchmark-specific attributes not in O3aDataCFG.
# whiten_padding = total padding (left + right) added to each slice,
# matching 2 * padding_length_in_s from BaseDataConfig.
data_cfg.whiten_padding = 2.0 * data_cfg.padding_length_in_s   # 4.0 s total
# Expected total sample count per slice element (signal + padding).
data_cfg.sample_length_in_num = data_cfg.padded_length_in_nsamples  # 32768
# tc prior from gwconfig.yaml (merger time from signal-window start).
data_cfg.tc_inject_lower = 11.0
data_cfg.tc_inject_upper = 11.2
# Signal-only duration without padding.
data_cfg.signal_length = data_cfg.sample_length_in_s  # 12.0 s

# --- Paths ---
TESTING_DIR = "/local/scratch/igr/nnarenraju/testing_month_D4_seeded"
EXPORT_DIR = os.path.join(RUN_DIR, "run_export")
BENCHMARK_DIR = os.path.join(EXPORT_DIR, "benchmark-mlgwsc1")
CHECKPOINT = os.path.join(EXPORT_DIR, "CHECKPOINTS", "epoch_19.pt")

os.makedirs(BENCHMARK_DIR, exist_ok=True)

# --- Inference hyperparameters ---
STEP_SIZE = 0.1          # step size in seconds between consecutive slices
TRIGGER_THRESHOLD = 0.0  # raw logit threshold; sigmoid(0) = 0.5
CLUSTER_THRESHOLD = 0.35 # cluster triggers within this many seconds
BATCH_SIZE = 256
NUM_WORKERS = 8
DEVICE = cfg.device
dtype = torch.float32


# ============================================================
# Slicer (adapted from sage/benchmark/mlgwsc1/mlgwsc1.py)
# Inlined here to avoid the broken import chain in that file.
# ============================================================

class Slicer:
    """Iterate over a single MLGWSC-1 HDF5 file in overlapping windows."""

    def __init__(self, infile, step_size, peak_offset, slice_length,
                 detectors=None, data_cfg=None):
        self.infile = infile
        self.step_size = step_size
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.data_cfg = data_cfg

        self.detectors = detectors
        if self.detectors is None:
            self.detectors = [
                self.infile[key] for key in list(self.infile.attrs["detectors"])
            ]
        self.keys = sorted(list(self.detectors[0].keys()), key=lambda x: int(x))
        self._determine_nslices()

    def _determine_nslices(self):
        self.n_slices = {}
        start = 0
        for ds_key in self.keys:
            ds = self.detectors[0][ds_key]
            dt = ds.attrs["delta_t"]
            index_step = int(self.step_size / dt)
            nsteps = int(
                (len(ds) - self.slice_length
                 - int(self.data_cfg.whiten_padding * self.data_cfg.sample_rate))
                // index_step
            )
            self.n_slices[ds_key] = {"start": start, "stop": start + nsteps, "len": nsteps}
            start += nsteps

    def __len__(self):
        return sum(v["len"] for v in self.n_slices.values())

    def _access_indices(self, index):
        assert index.step is None or index.step == 1
        ret = {}
        start, stop = index.start, index.stop
        for key in self.keys:
            cs, ce = self.n_slices[key]["start"], self.n_slices[key]["stop"]
            if cs <= start < ce:
                ret[key] = slice(start, min(stop, ce))
                start = ret[key].stop
        return ret

    def _generate_data(self, key, index):
        dt = 1.0 / 2048.0
        index_step = int(self.step_size / dt)
        extra = int(self.data_cfg.whiten_padding * self.data_cfg.sample_rate)

        outer_sidx = (index.start - self.n_slices[key]["start"]) * index_step
        outer_eidx = (
            (index.stop - self.n_slices[key]["start"]) * index_step
            + self.slice_length + extra
        )
        rawdata = [det[key][outer_sidx:outer_eidx] for det in self.detectors]

        times = (
            (self.detectors[0][key].attrs["start_time"] + outer_sidx * dt)
            + index_step * dt * np.arange(index.stop - index.start)
            + self.peak_offset
        )

        N = index.stop - index.start
        total = self.slice_length + extra
        data = np.zeros((N, len(rawdata), total))
        for d, rawdat in enumerate(rawdata):
            for i in range(N):
                s = i * index_step
                e = s + total
                ts = pycbc.types.TimeSeries(rawdat[s:e], delta_t=dt)
                data[i, d, :] = ts.numpy()

        return data, times

    def __getitem__(self, index):
        is_single = isinstance(index, int)
        if is_single:
            if index < 0:
                index = len(self) + index
            index = slice(index, index + 1)

        access = self._access_indices(index)
        data, times = [], []
        for key, idxs in access.items():
            d, t = self._generate_data(key, idxs)
            data.append(d)
            times.append(t)

        data = np.concatenate(data)
        times = np.concatenate(times)

        if is_single:
            return data[0], times[0]
        return data, times


class RawTorchSlicer(Slicer, torch.utils.data.Dataset):
    """Wraps Slicer and returns raw float32 tensors for the DataLoader."""

    def __init__(self, *args, **kwargs):
        torch.utils.data.Dataset.__init__(self)
        Slicer.__init__(self, *args, **kwargs)

    def __getitem__(self, index):
        raw, t = Slicer.__getitem__(self, index)
        return torch.from_numpy(raw.copy()).float(), torch.tensor(t, dtype=torch.float64)


# ============================================================
# Model
# ============================================================

from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic

print("Building model ...")
model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
    frontend_filters=32,
    frontend_kernel=64,
    backend_resnet_size=50,
    norm_type="groupnorm",   # MUST match training (train_hard.py); else weights load into wrong norm
)

print(f"Loading weights from {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"]
# torch.compile prefixes keys with '_orig_mod.'; strip it for uncompiled inference.
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
model.to(device=DEVICE, dtype=dtype)
print(f"Model loaded (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

# ============================================================
# Preprocessor: FiducialWhitening + MultirateSampler
# ============================================================

from sage.dsp.whiten import FiducialWhitening
from sage.dsp.multirate_sampling import MultirateSampler, DyadicPyramidBinning
from sage.data.waveform import read_from_config
from sage.core.graph import Preprocessor

gwconfig_path = os.path.join(RUN_DIR, "gwconfig.yaml")
param_sampler = read_from_config(gwconfig_path, seed=150914)
bounds = param_sampler.bounds

whitener = FiducialWhitening()
dyadic_binning = DyadicPyramidBinning(bounds)
mrsampler = MultirateSampler(binning_method=dyadic_binning)
processor = Preprocessor([whitener, mrsampler])
processor.eval()
processor.to(device=DEVICE)
print("Preprocessor (FiducialWhitening + MultirateSampler) ready.")

# ============================================================
# Trigger clustering
# ============================================================

def get_clusters(triggers, cluster_threshold=0.35):
    clusters = []
    for trig in triggers:
        new_t = trig[0]
        if not clusters or (new_t - clusters[-1][-1][0]) > cluster_threshold:
            clusters.append([trig])
        else:
            clusters[-1].append(trig)

    print(f"  Clustering produced {len(clusters)} independent triggers.")

    cluster_times, cluster_values, cluster_timevars = [], [], []
    for cluster in clusters:
        times = [t[0] for t in cluster]
        values = np.array([t[1] for t in cluster])
        idx = np.argmax(values)
        cluster_times.append(times[idx])
        cluster_values.append(values[idx])
        cluster_timevars.append(0.35)

    return np.array(cluster_times), np.array(cluster_values), np.array(cluster_timevars)


# ============================================================
# Inference loop
# ============================================================

def run_inference(inputfile, outputfile):
    # Signal-only slice length: 12 s × 2048 Hz = 24576 samples
    slice_length = int(data_cfg.signal_length * data_cfg.sample_rate)

    # peak_offset: GPS time offset from each slice's raw start to expected merger.
    # tc is measured from the signal window start (after left_padding = whiten_padding/2).
    # So merger GPS time = T_slice_start + left_padding + tc_mid.
    tc_mid = (data_cfg.tc_inject_lower + data_cfg.tc_inject_upper) / 2.0
    peak_offset = tc_mid + data_cfg.whiten_padding / 2.0  # = 11.1 + 2.0 = 13.1 s

    triggers = []

    with h5py.File(inputfile, "r") as infile:
        slicer = RawTorchSlicer(
            infile,
            step_size=STEP_SIZE,
            peak_offset=peak_offset,
            slice_length=slice_length,
            data_cfg=data_cfg,
        )
        print(f"  Total slices: {len(slicer):,}")

        data_loader = torch.utils.data.DataLoader(
            slicer,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

        max_trigger = torch.tensor(-1e9, device=DEVICE)

        for raw_batch, slice_times in tqdm(
            data_loader, desc=f"  {os.path.basename(inputfile)}"
        ):
            with torch.inference_mode():
                x = raw_batch.to(device=DEVICE, dtype=dtype)  # (B, D, T_padded)

                # Time domain → frequency domain.
                # norm="forward" (divides by N) matches MemmapNoiseSampler._read_batch,
                # which uses rfft(x, norm="forward") before passing to FiducialWhitening.
                x_fd = torch.fft.rfft(x, dim=-1, norm="forward")  # (B, D, F) complex64

                # Preprocessor runs outside autocast (complex FD input is not cast)
                x_proc = processor(x_fd)   # whiten + multirate → (B, D, L)

                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if cfg.autocast
                    else nullcontext()
                ):
                    out = model(x_proc)

            # out = (ranking_statistic, point_estimates)
            # ranking_statistic: (B, 1) raw logit
            raw_values = out[0].squeeze(1).float()  # (B,) on GPU
            trigger_mask = (raw_values > TRIGGER_THRESHOLD).cpu()  # (B,) on CPU

            max_trigger = torch.max(max_trigger, raw_values.max())

            if trigger_mask.any():
                raw_cpu = raw_values.cpu()
                for t, val in zip(
                    slice_times[trigger_mask].tolist(),
                    raw_cpu[trigger_mask].tolist(),
                ):
                    triggers.append([t, val])

        print(f"  Max trigger logit: {max_trigger.item():.4f}")
        print(f"  Triggers above threshold ({TRIGGER_THRESHOLD}): {len(triggers):,}")

    if len(triggers) == 0:
        raise ValueError(
            f"No triggers found above threshold {TRIGGER_THRESHOLD}. "
            "Consider lowering TRIGGER_THRESHOLD."
        )

    _t = np.array(triggers)
    print(f"  Logit range: max={_t[:,1].max():.4f}, min={_t[:,1].min():.4f}")

    times, stats, variances = get_clusters(triggers, CLUSTER_THRESHOLD)

    with h5py.File(outputfile, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("stat", data=stats)
        f.create_dataset("var", data=variances)

    print(f"  Clustered events saved: {len(times)}")
    print(f"  Output: {outputfile}")
    return times, stats, variances


# ============================================================
# Run
# ============================================================

print("\n=== MLGWSC-1 Benchmark: Background ===")
bg_input = os.path.join(TESTING_DIR, "background.hdf")
bg_output = os.path.join(BENCHMARK_DIR, "bg_events.hdf")
run_inference(bg_input, bg_output)

print("\n=== MLGWSC-1 Benchmark: Foreground ===")
fg_input = os.path.join(TESTING_DIR, "foreground.hdf")
fg_output = os.path.join(BENCHMARK_DIR, "fg_events.hdf")
run_inference(fg_input, fg_output)

print(f"\nInference complete. Results in: {BENCHMARK_DIR}")
