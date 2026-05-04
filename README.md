## Sage - Gravitational Wave Detection using Machine Learning

<p align="center">
  <img src="https://github.com/user-attachments/assets/ceabdb59-2847-45e6-a618-2153278049d0" alt="SAGE logo" width="400"/>
</p>

[![DOI](https://zenodo.org/badge/482025216.svg)](https://doi.org/10.5281/zenodo.17290133)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Identifying and Mitigating Machine Learning Biases for the Gravitational Wave Detection Problem](https://arxiv.org/abs/2501.13846)

**Abstract**

Matched-filtering is a long-standing technique for the optimal detection of known signals in stationary Gaussian noise. However, it has known departures from optimality when operating on unknown signals in real noise and suffers from computational inefficiencies in its pursuit to near-optimality. A compelling alternative that has emerged in recent years to address this problem is deep learning. Although it has shown significant promise when applied to the search for gravitational-waves in detector noise, we demonstrate the existence of a multitude of learning biases that hinder generalisation and detection performance. Our work identifies the sources of a set of 11 interconnected biases present in the supervised learning of the gravitational-wave detection problem, and contributes mitigation tactics and training strategies to concurrently address them. We introduce, Sage, a machine-learning based binary black hole search pipeline. We evaluate our pipeline on the injection study presented in the Machine Learning Gravitational-Wave Search Challenge and show that Sage detects ~11.2% more signals than the benchmark PyCBC analysis at a false alarm rate of one per month in O3a noise. Moreover, we also show that it can detect ~48.29% more signals than the previous best performing machine-learning pipeline on the same dataset. We empirically prove that our pipeline has the capability to effectively handle out-of-distribution noise power spectral densities and reject non-Gaussian transient noise artefacts. By studying machine-learning biases and conducting empirical investigations to understand the reasons for performance improvement/degradation, we aim to address the need for interpretability of machine-learning methods for gravitational-wave detection.

---

## Installation

> **Note:** These are local installation instructions. Sage will be available as a proper PyPI package soon.

```bash
git clone https://github.com/nnarenraju/sage.git
cd sage
pip install -e .
```

**Core dependencies** (installed automatically via `setup.py`): PyTorch ≥ 2.0, h5py, numpy, scipy, astropy, tqdm, pyyaml.

**Optional dependencies** for data acquisition and benchmarking:
```bash
pip install pycbc lalsuite gwpy gwosc
```

**Hardware**: A CUDA-capable GPU is strongly recommended for on-the-fly waveform generation and training.

---

## Repository Structure

```
sage/
├── sage/
│   ├── architecture/       # Neural network components
│   │   ├── backend/        #   2D/3D ResNet with CBAM attention
│   │   ├── frontend/       #   Multi-scale 1D CNN (per detector)
│   │   ├── network/        #   Assembled full networks
│   │   ├── custom_losses/  #   BCE-based loss functions
│   │   └── zoo/            #   Cross-attention modules
│   ├── core/               # Config, logging, constants, interpolation
│   ├── data/
│   │   ├── noise/          #   Real noise samplers, hard mining, glitch oversampling
│   │   ├── psd/            #   PSD generation and loading
│   │   ├── primer/         #   Data download utilities
│   │   └── waveform/       #   GW parameter sampling, projection, SNR rescaling
│   ├── dsp/                # FFT, whitening, multirate sampling, Welch PSD
│   ├── exec/               # Top-level pipeline orchestration (SageDirector)
│   ├── factory/            # Training/validation loops, schedulers, callbacks
│   ├── plotting/           # Diagnostic visualisation (ROC, efficiency, PE, …)
│   ├── presets/            # Pre-built config presets for common experiments
│   └── utils/              # Checkpointing, timing, Condor utilities
├── runs/                   # Run scripts for specific experiments
├── repro/                  # Reproducibility scripts for paper results
└── docs/                   # Sphinx/ReadTheDocs documentation source
```

---

## Quick Start

Reproducibility scripts and a walkthrough notebook are provided in [`repro/`](repro/). Start with [`repro/start_here.ipynb`](repro/start_here.ipynb).

---

## Documentation

Full API documentation is available at **[sage-gw.readthedocs.io](https://sage-gw.readthedocs.io/en/latest/)**.

---

## Testing

A minimal smoke test exercises the waveform pipeline end-to-end:

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

---

## Contributing

Contributions are welcome.  Please open an issue first to discuss the proposed change, then submit a pull request against the `main` branch.

1. Fork the repository and create a feature branch from `main`.
2. Write or update tests to cover the new behaviour.
3. Ensure all existing syntax checks pass: `python -m py_compile sage/**/*.py`.
4. Update docstrings (NumPy style) and `CHANGELOG.md` if applicable.
5. Open a pull request with a clear description of the motivation and approach.

---

## Cite
If you found this work useful in your research, please consider citing:

```
@misc{nagarajan2025,
      title={Identifying and Mitigating Machine Learning Biases for the Gravitational-wave Detection Problem}, 
      author={Narenraju Nagarajan and Christopher Messenger},
      year={2025},
      eprint={2501.13846},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2501.13846}, 
}
```
---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

We appreciate the useful comments from Thomas Dent and Nikolaos Stergioulas on our paper. NN wishes to acknowledge and appreciate the support of Joseph Bayley, Michael Williams and Christian Chapman-Bird. We would also like to extend our sincere gratitude to the PHAS-ML group members from the University of Glasgow, for their fruitful weekly meetings. NN is supported by the College Scholarship offered by the School of Physics and Astronomy (2021-2025), University of Glasgow. CM is supported by STFC grant ST/Y004256/1. This material is based upon work supported by NSF's LIGO Laboratory, a major facility fully funded by the National Science Foundation.
