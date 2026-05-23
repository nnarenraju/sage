## Sage - Gravitational Wave Detection with Machine Learning

<p align="center">
  <img src="https://github.com/user-attachments/assets/ceabdb59-2847-45e6-a618-2153278049d0" alt="SAGE logo" width="400"/>
</p>

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17290133-blue)](https://doi.org/10.5281/zenodo.17290133)
[![CI](https://github.com/nnarenraju/sage/actions/workflows/ci.yml/badge.svg)](https://github.com/nnarenraju/sage/actions/workflows/ci.yml)
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

Matched-filtering is a long-standing technique for the optimal detection of known signals in stationary Gaussian noise. However, it has known departures from optimality when operating on unknown signals in real noise and suffers from computational inefficiencies in its pursuit to near-optimality. A compelling alternative that has emerged in recent years to address this problem is deep learning. Although it has shown significant promise when applied to the search for gravitational-waves in detector noise, we demonstrate the existence of a multitude of learning biases that hinder generalisation and detection performance. Our work identifies the sources of a set of 11 interconnected biases present in the supervised learning of the gravitational-wave detection problem, and contributes mitigation tactics and training strategies to concurrently address them. We introduce, Sage, a machine-learning based binary black hole search pipeline. We evaluate our pipeline on the injection study presented in the Machine Learning Gravitational-Wave Search Challenge and show that Sage detects ~11.2% more signals than the benchmark PyCBC analysis at a false alarm rate of one per month in O3a noise. Moreover, we also show that it can detect ~48.29% more signals than the previous best performing machine-learning pipeline on the same dataset. We empirically prove that our pipeline has the capability to effectively handle out-of-distribution noise power spectral densities and reject non-Gaussian transient noise artefacts. By studying machine-learning biases and conducting empirical investigations to understand the reasons for performance improvement/degradation, we aim to address the need for interpretability of machine-learning methods for gravitational-wave detection.

The repository contains the research code used for the Sage pipeline, including:

- Binary black hole waveform generation and detector projection.
- Real-noise sampling, PSD handling, whitening, and preprocessing utilities.
- Time-domain multirate sampling and frequency-domain multibanding utilities.
- Neural network architectures and training loops.
- Reproducibility notebooks and run scripts for paper-style experiments.
- Diagnostic plotting tools for ranking statistics, efficiency curves, ROC
  curves, and parameter studies.

All modules are GPU compatible.

---

## Installation

Sage is currently intended for local editable installs.

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
