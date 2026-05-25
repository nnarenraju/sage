## Sage - Gravitational Wave Detection with Machine Learning

<p align="center">
  <img src="docs/_static/sage_logo.png" alt="SAGE logo" width="400"/>
</p>

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20375078-blue)](https://doi.org/10.5281/zenodo.20375078)
[![ASCL](https://img.shields.io/badge/ascl-4712-blue.svg?colorB=262255)](https://www.ascl.net/code/v/4712)
[![CI](https://github.com/nnarenraju/sage/actions/workflows/ci.yml/badge.svg)](https://github.com/nnarenraju/sage/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/nnarenraju/sage/branch/main/graph/badge.svg?token=RLAAMEZEZ6)](https://codecov.io/github/nnarenraju/sage)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Sage is a machine-learning based search pipeline for compact binary coalescence
signals in gravitational-wave detector data. It includes tools for realistic
noise handling, waveform generation, detector projection, whitening, signal
compression, training, validation, and diagnostic studies.

The methods are described in:

[Identifying and Mitigating Machine Learning Biases for the Gravitational Wave Detection Problem](https://journals.aps.org/prd/abstract/10.1103/zwj9-ycyz)

**Abstract**

Sage is a complete, end-to-end machine-learning pipeline for gravitational-wave (GW) compact binary coalescence (CBC) detection. Training operates entirely on-the-fly — no pre-computed datasets required — with waveforms and noise windows generated per batch to eliminate data-reuse biases. Sage systematically identifies and mitigates 11 interconnected supervised-learning biases that degrade detection performance and generalisation. On the Machine Learning Gravitational-Wave Search Challenge injection study, Sage detects ~11.2% more signals than benchmark PyCBC matched-filtering and ~48.3% more than the previous best-performing ML pipeline at a false alarm rate of one per month, while remaining robust to out-of-distribution PSDs and non-Gaussian transient artefacts.

The repository contains the research code used for the Sage pipeline, including:

- Automated download and preparation of GWOSC data releases, segments, and PSDs.
- Realistic noise simulation: real strain, coloured, recoloured, and glitch-injected noise.
- Binary black hole waveform generation (IMRPhenomD/PhenomPv2), multi-detector projection, and SNR utilities.
- Signal processing: whitening, inverse spectrum truncation, time-domain multirate sampling, frequency-domain multibanding, and prior-median heterodyning.
- Hard noise mining for low-FAR robustness: brute-force, MAP-Elites, and Cross-Entropy Method strategies.
- Modular neural network architectures with interchangeable frontends, backends, and attention mechanisms.
- Training loops with on-the-fly data generation, schedulers, callbacks, and checkpointing.
- Google Colab tutorials, reproducibility notebooks, and run scripts for paper-style experiments.
- Diagnostic plotting tools for ranking statistics, efficiency curves, ROC curves, and parameter studies.

All modules are optimised for CPU and GPU (PyTorch compile-friendly) usage.

---

## Installation

Sage is currently intended for local editable installs.

**Option A — conda (recommended for GPU clusters)**

The [`utils/environment.yml`](utils/environment.yml) file defines the full
conda environment (Python 3.11, LALSuite, PyCBC, GWpy, and JupyterLab).
[`utils/create_env.sh`](utils/create_env.sh) automates the three-step setup:
conda packages → PyTorch CUDA wheel → editable Sage install.

```bash
git clone https://github.com/nnarenraju/sage.git
cd sage/utils
bash create_env.sh
conda activate sage
```

**Option B — pip only**

```bash
git clone https://github.com/nnarenraju/sage.git
cd sage
python -m pip install -r requirements.txt
python -m pip install -e .
```

PyTorch installation can depend on your CUDA version. If needed, install the
appropriate PyTorch build first using the command from
[pytorch.org](https://pytorch.org/get-started/locally/), then install the
remaining requirements.

A CUDA-capable GPU is strongly recommended for on-the-fly waveform generation,
training, and large injection studies.

---

## Repository Structure

```
sage/
├── sage/
│   ├── architecture/       # Frontend, backend, attention, and full networks
│   ├── benchmark/          # Benchmark integrations and comparison utilities
│   ├── core/               # Config, logging, constants, interpolation
│   ├── data/
│   │   ├── noise/          # Real noise samplers, hard mining, glitch handling
│   │   ├── primer/         # Data download and preparation utilities
│   │   ├── psd/            # PSD generation and loading
│   │   └── waveform/       # Parameter sampling, waveforms, projection, SNR
│   ├── dsp/                # FFT, whitening, PSDs, multirate, multibanding
│   ├── exec/               # Pipeline orchestration
│   ├── factory/            # Training, validation, schedulers, callbacks
│   ├── plotting/           # Diagnostic and publication plotting
│   ├── presets/            # Pre-built config presets
│   └── utils/              # Checkpointing, timing, Condor utilities
├── runs/                   # Run scripts for specific experiments
├── repro/                  # Reproducibility notebooks and configuration
├── notebooks/              # Exploratory notebooks
├── tests/                  # Lightweight tests and smoke checks
└── docs/                   # Sphinx/ReadTheDocs documentation source
```

---

## Colab Tutorials

No local install needed — run Sage on a free Google Colab T4 GPU:

| | Notebook | |
|---|---|---|
| 1 | Signal generation with IMRPhenomD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/01_signal_generation.ipynb) |
| 2 | Realistic data simulation and whitening | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/02_data_simulation.ipynb) |
| 3 | Training and evaluating a GW detector | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/03_training_and_evaluation.ipynb) |

---

## Quick Start

Start with [`repro/start_here.ipynb`](repro/start_here.ipynb), which walks
through the main Sage workflow used by the reproducibility scripts.

The run-specific scripts live under [`runs/`](runs/), and shared configuration
presets live under [`sage/presets/`](sage/presets/).

---

## Documentation

Full API documentation is available at **[sage-gw.readthedocs.io](https://sage-gw.readthedocs.io/en/latest/)**.

---

## Testing

A minimal smoke test for configuration registration:

```bash
python -c "
from sage.core.config import register_configs
from sage.presets.data_configs import Default as data_cfg
from sage.presets.configs import DefaultConfig as cfg
register_configs(cfg, data_cfg)
print('Config registration: OK')
"
```

Individual module tests can be run with pytest (where present):

```bash
pytest tests/ -v
```

For a broad syntax check:

```bash
python -m py_compile $(find sage -name '*.py')
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss substantial
changes, then submit a pull request against the `main` branch.

1. Fork the repository and create a feature branch from `main`.
2. Add or update tests for behaviour that changes.
3. Run the relevant tests and syntax checks.
4. Update documentation, docstrings, and `CHANGELOG.md` when applicable.
5. Open a pull request with a clear description of the motivation and approach.

---

## Citation

If you use Sage in your research, please cite:

```bibtex
@article{sage,
  title = {Identifying and mitigating machine-learning biases for the gravitational-wave detection problem},
  author = {Nagarajan, Narenraju and Messenger, Christopher},
  journal = {Phys. Rev. D},
  volume = {112},
  issue = {10},
  pages = {103002},
  numpages = {40},
  year = {2025},
  month = {Nov},
  publisher = {American Physical Society},
  doi = {10.1103/zwj9-ycyz},
  url = {https://link.aps.org/doi/10.1103/zwj9-ycyz}
}
```
---

## License

Sage is released under the [GNU General Public License v3.0](LICENSE).

---

## Acknowledgements

We appreciate the useful comments from Thomas Dent and Nikolaos Stergioulas on our paper. NN wishes to acknowledge and appreciate the support of Joseph Bayley, Michael Williams and Christian Chapman-Bird. We would also like to extend our sincere gratitude to the PHAS-ML group members from the University of Glasgow, for their fruitful weekly meetings. NN is supported by the College Scholarship offered by the School of Physics and Astronomy (2021-2025), University of Glasgow. CM is supported by STFC grant ST/Y004256/1. This material is based upon work supported by NSF's LIGO Laboratory, a major facility fully funded by the National Science Foundation.
