# sage/diagnostics

Standalone, CPU-only diagnostics for the Sage data + signal pipeline. Each file
is a runnable script named `diagnose_<thing>.py` that **prints the numbers** and
**saves plots** to `sage/diagnostics/plots/` (git-ignored). This suite grows over
time — add new checks as `diagnose_*.py` following the same convention.

## Running

Run directly with the sage python, on a login/CPU node, with the server set:

```bash
SAGE_SERVER=jarvis /work/nagarajan/conda_envs/sage/bin/python \
    sage/diagnostics/diagnose_prior.py
```

Each script `chdir`s into `runs/o3b` to read `gwconfig.yaml` + the fiducial PSDs,
forces `device="cpu"`, and writes outputs to `plots/`. To point a diagnostic at a
different run, change the `os.chdir(...)` target near the top of the file.

## Current diagnostics

| script | what it checks |
|---|---|
| `diagnose_prior.py` | full parameter prior (masses, spins, distance, tc, sky) |
| `diagnose_signal_pipeline.py` | SNR distribution, tc placement, multirate alignment, amplitude |
| `diagnose_amplitude.py` | whitening TD scale vs optimal-SNR estimator consistency |
| `diagnose_snr_vs_pycbc.py` | optimal SNR vs PyCBC `sigmasq` (padded 1/16 == correct) |
| `diagnose_tc_offset.py` | raw vs whitened \|h\|-peak vs injected tc; offset-vs-mass |
| `diagnose_tc_offset_drivers.py` | tc offset correlated vs inclination / precession / mass |
| `diagnose_tc_vs_pycbc.py` | Sage vs PyCBC merger time (coalescence reference match) |
| `diagnose_perdet_tc.py` | per-detector tc decomposition (light-travel-time) |
| `diagnose_noise_datascan.py` | on-disk O3b noise: amplitude, NaN/gaps, glitches, ASD |

## Adding a new diagnostic

Copy the header of any existing script (sys.path + `os.chdir(runs/o3b)` + the
inline CPU `C`/`DC` config + `register_configs`), do the check, `print` the key
numbers, and `savefig` into `plots/`. Name it `diagnose_<thing>.py`.
