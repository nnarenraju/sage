# Tests

[![codecov](https://codecov.io/github/nnarenraju/sage/branch/main/graph/badge.svg?token=RLAAMEZEZ6)](https://codecov.io/github/nnarenraju/sage)

## Coverage

![Coverage graph](https://codecov.io/github/nnarenraju/sage/graphs/tree.svg?token=RLAAMEZEZ6)

## Structure

Lightweight unit tests for core Sage behaviour, runnable on CPU-only environments
without optional dependencies (pycbc, lal, astropy).

| File | Module under test |
|---|---|
| `test_compiled.py` | `sage` — scatter-add equivalence |
| `test_core_conversions.py` | `sage.core.conversions` — mass/time/sample conversions |
| `test_core_math.py` | `sage.core.math` — normalise, standardise, rotation matrices |
| `test_core_utils.py` | `sage.core.utils`, `sage.core.constants` — helpers and physical constants |
| `test_dsp_multibanding.py` | `sage.dsp.multibanding` — frequency bands, layout, compressor |
| `test_dsp_heterodyning.py` | `sage.dsp.heterodyning` — heterodyne application, residual chirp time |
| `test_dsp_fft.py` | `sage.dsp.fft` — FFT wrapper, shape, correctness |
| `test_waveform_conversions.py` | `sage.data.waveform.conversions` — chirp mass, mass ratio, chirp distance |
| `test_waveform_taper.py` | `sage.data.waveform.taper` — frequency-domain tapers |
| `test_data_hard_mining.py` | `sage.data.noise.hard_mining` — buffer and streaming top-k |

## Running

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=sage --cov-branch --cov-report=term-missing
```
