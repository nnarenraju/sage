## Sage - Gravitational Wave Detection using Machine Learning

<p align="center">
  <img src="https://github.com/user-attachments/assets/ceabdb59-2847-45e6-a618-2153278049d0" alt="SAGE logo" width="400"/>
</p>

[![DOI](https://zenodo.org/badge/482025216.svg)](https://doi.org/10.5281/zenodo.17290133)

[Identifying and Mitigating Machine Learning Biases for the Gravitational Wave Detection Problem](https://arxiv.org/abs/2501.13846)

**Abstract**

Matched-filtering is a long-standing technique for the optimal detection of known signals in stationary Gaussian noise. However, it has known departures from optimality when operating on unknown signals in real noise and suffers from computational inefficiencies in its pursuit to near-optimality. A compelling alternative that has emerged in recent years to address this problem is deep learning. Although it has shown significant promise when applied to the search for gravitational-waves in detector noise, we demonstrate the existence of a multitude of learning biases that hinder generalisation and detection performance. Our work identifies the sources of a set of 11 interconnected biases present in the supervised learning of the gravitational-wave detection problem, and contributes mitigation tactics and training strategies to concurrently address them. We introduce, Sage, a machine-learning based binary black hole search pipeline. We evaluate our pipeline on the injection study presented in the Machine Learning Gravitational-Wave Search Challenge and show that Sage detects ~11.2% more signals than the benchmark PyCBC analysis at a false alarm rate of one per month in O3a noise. Moreover, we also show that it can detect ~48.29% more signals than the previous best performing machine-learning pipeline on the same dataset. We empirically prove that our pipeline has the capability to effectively handle out-of-distribution noise power spectral densities and reject non-Gaussian transient noise artefacts. By studying machine-learning biases and conducting empirical investigations to understand the reasons for performance improvement/degradation, we aim to address the need for interpretability of machine-learning methods for gravitational-wave detection.

---

## Installation

```bash
git clone https://github.com/nnarenraju/sage.git
cd sage
pip install -e .
```

**Dependencies** (installed automatically): PyTorch ≥ 2.0, PyCBC, h5py, numpy, scipy, astropy, tqdm, pyyaml.

---

## Repository Structure

```
sage/
├── sage/
│   ├── architecture/       # Neural network components
│   │   ├── backend/        #   2D/3D ResNet with CBAM attention
│   │   ├── frontend/       #   Multi-scale 1D CNN (per detector)
│   │   ├── network/        #   Assembled full networks
│   │   ├── custom_losses/  #   BCEWithFARLoss and related
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
└── docs/                   # Sphinx/ReadTheDocs documentation source
```

---

## Key Design Choices

| Feature | Detail |
|---------|--------|
| **On-the-fly training** | No pre-generated dataset — signals and noise are drawn per batch |
| **Noise** | Real LIGO O3b strain via memory-mapped async prefetch |
| **Signal injection** | Random batch positions; SNR drawn from prior distribution |
| **Preprocessing** | FD whitening + dyadic multirate decimation (physics-driven bin layout) |
| **Loss** | BCE + heteroscedastic regression + coupling + pAUC + focal (6-component) |
| **Hard mining** | Online hard background buffer + worst-missed-signal replay |
| **Glitch robustness** | GravitySpy-aligned glitch windows oversampled 10% of each batch |
| **Domain adaptation** | Stochastic O3b → O3a PSD recolouring during training |
| **Compilation** | `torch.compile(mode="max-autotune")` for the inner training loop |

---

## Quick Start

```python
from sage.core.config import register_configs
from sage.data.waveform.sampler import read_from_config
from sage.data.noise.real_noise import MemmapNoiseSampler

# 1. Register configs (must be called once)
register_configs(cfg, data_cfg)

# 2. Build waveform sampler from a YAML prior
param_sampler = read_from_config("priors/bbh_o3.yaml", seed=42)

# 3. Build noise sampler
noise_sampler = MemmapNoiseSampler(data_cfg)

# 4. Draw a batch
params = param_sampler(batch_size)          # (B, num_params)
noise, noise_targets = noise_sampler()      # (B, D, T)
```

See `runs/o3b_hardmining/train.py` for a complete training script.

---

## Documentation

Full API documentation is available at ReadTheDocs (auto-built from docstrings):

```bash
cd docs
pip install -r requirements.txt
make html
# open build/html/index.html
```

---

### Cite
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
Will be updated after publication.

### Acknowledgements

We appreciate the useful comments from Thomas Dent and Nikolaos Stergioulas on our paper. NN wishes to acknowledge and appreciate the support of Joseph Bayley, Michael Williams and Christian Chapman-Bird. We would also like to extend our sincere gratitude to the PHAS-ML group members from the University of Glasgow, for their fruitful weekly meetings. NN is supported by the College Scholarship offered by the School of Physics and Astronomy (2021-2025), University of Glasgow. CM is supported by STFC grant ST/Y004256/1. This material is based upon work supported by NSF’s LIGO Laboratory, a major facility fully funded by the National Science Foundation.
