.. Sage documentation master file

Sage — Gravitational Wave Detection Pipeline
============================================

.. image:: https://zenodo.org/badge/482025216.svg
   :target: https://doi.org/10.5281/zenodo.17290133
   :alt: DOI

**Sage** is a machine-learning based binary black-hole search pipeline.

Reference paper:
`Identifying and Mitigating Machine Learning Biases for the Gravitational-Wave Detection Problem
<https://arxiv.org/abs/2501.13846>`_ — Nagarajan & Messenger (2025).

Abstract
--------

Matched-filtering is a long-standing technique for the optimal detection of
known signals in stationary Gaussian noise. However, it has known departures
from optimality when operating on unknown signals in real noise and suffers
from computational inefficiencies in its pursuit to near-optimality. A
compelling alternative that has emerged in recent years to address this problem
is deep learning. Although it has shown significant promise when applied to the
search for gravitational-waves in detector noise, we demonstrate the existence
of a multitude of learning biases that hinder generalisation and detection
performance. Our work identifies the sources of a set of 11 interconnected
biases present in the supervised learning of the gravitational-wave detection
problem, and contributes mitigation tactics and training strategies to
concurrently address them. We introduce, Sage, a machine-learning based binary
black hole search pipeline. We evaluate our pipeline on the injection study
presented in the Machine Learning Gravitational-Wave Search Challenge and show
that Sage detects ~11.2% more signals than the benchmark PyCBC analysis at a
false alarm rate of one per month in O3a noise. Moreover, we also show that it
can detect ~48.29% more signals than the previous best performing
machine-learning pipeline on the same dataset. We empirically prove that our
pipeline has the capability to effectively handle out-of-distribution noise
power spectral densities and reject non-Gaussian transient noise artefacts. By
studying machine-learning biases and conducting empirical investigations to
understand the reasons for performance improvement/degradation, we aim to
address the need for interpretability of machine-learning methods for
gravitational-wave detection.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   self

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   autoapi/sage/index


Installation
------------

.. note::

   These are local installation instructions. Sage will be available as a
   proper PyPI package soon.

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0, CUDA (optional but recommended).

Clone and install in editable mode::

    git clone https://github.com/nnarenraju/sage.git
    cd sage
    pip install -e .

Additional dependencies for data acquisition and benchmarking::

    pip install pycbc lalsuite gwpy gwosc h5py


Quick Start
-----------

Reproducibility scripts and a walkthrough notebook are provided in the
`repro/ <https://github.com/nnarenraju/sage/tree/main/repro>`_ directory of
the repository. Start with ``repro/start_here.ipynb``.


Overview
--------

Architecture
~~~~~~~~~~~~

Sage uses a **two-stage neural network**:

1. **Frontend** — per-detector multi-scale 1D CNN
   (:class:`~sage.architecture.frontend.mscnn1d.ConvBlock`) that extracts
   temporal features across a wide range of time scales using five parallel
   convolutions at different kernel sizes.

2. **Backend** — 2D ResNet with CBAM (Convolutional Block Attention Module)
   attention
   (:class:`~sage.architecture.backend.resnet2d_cbam.ResNet`) that processes
   the stacked per-detector feature maps and produces a compact feature vector.

The network head outputs:

* A **ranking statistic** (raw logit for BCE classification).
* **Heteroscedastic point estimates** (mean + log-variance) for each
  regression target (e.g. chirp mass, coalescence time).

Training Pipeline
~~~~~~~~~~~~~~~~~

Training is performed on-the-fly (OTF) — no fixed dataset is pre-generated.
At each iteration:

1. Gravitational-wave signals are sampled from the prior
   (:class:`~sage.data.waveform.sampler.DistributionSampler`), projected
   onto detectors, and SNR-rescaled.
2. Real LIGO noise windows are asynchronously fetched from a memory-mapped
   file (:class:`~sage.data.noise.real_noise.MemmapNoiseSampler`).
3. Signals are injected at random positions in the noise batch.
4. The pipeline applies FD whitening
   (:class:`~sage.dsp.whiten.FiducialWhitening`) and dyadic multirate
   sampling (:class:`~sage.dsp.multirate_sampling.MultirateSampler`).
5. The loss is computed using
   :class:`~sage.architecture.custom_losses.loss_functions.BCEWithPEsigmaLoss`
   — BCE classification combined with heteroscedastic regression and coupling
   regularisation for simultaneous detection and parameter estimation.

Hard-noise mining (:class:`~sage.data.noise.hard_mining.HardSampleMiner`)
and glitch oversampling
(:class:`~sage.data.noise.glitch_sampler.GlitchOversampledNoiseSampler`)
are used to address dataset imbalances and improve robustness to transient
noise artefacts.

Waveform Generation
~~~~~~~~~~~~~~~~~~~

Two GPU-native batched approximants are provided:

* :class:`~sage.data.waveform.approximants.IMRPhenomD.IMRPhenomD` — aligned-spin
  frequency-domain waveforms, ``torch.compile``-compatible (``GRAPH_READY = True``).
* :class:`~sage.data.waveform.approximants.IMRPhenomPv2.IMRPhenomPv2` — precessing-spin
  extension via the PhenomPv2 "twist-up" formalism.

Waveform parameters are drawn from a YAML-configured prior using
:class:`~sage.data.waveform.sampler.DistributionSampler`.

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :mod:`sage.architecture`
     - Network components: frontend CNN, backend ResNet-CBAM, loss functions
   * - :mod:`sage.core`
     - Configuration management, logging, constants, interpolation, math utilities
   * - :mod:`sage.data`
     - Waveform generation, noise sampling, PSD handling, data primitives
   * - :mod:`sage.dsp`
     - Digital signal processing: FFT, whitening, multirate sampling, Welch PSD
   * - :mod:`sage.exec`
     - Execution orchestration: ``SageDirector``, config/data/export handlers
   * - :mod:`sage.factory`
     - Training/validation loops, callbacks, schedulers, compile manager
   * - :mod:`sage.plotting`
     - Diagnostic plots: ROC curves, efficiency, parameter recovery, loss curves
   * - :mod:`sage.presets`
     - Pre-built configuration presets for common training scenarios
   * - :mod:`sage.utils`
     - Checkpointing, timing utilities, Condor job submission

Citing Sage
-----------

If you found this work useful in your research, please cite::

    @misc{nagarajan2025,
          title={Identifying and Mitigating Machine Learning Biases for the
                 Gravitational-wave Detection Problem},
          author={Narenraju Nagarajan and Christopher Messenger},
          year={2025},
          eprint={2501.13846},
          archivePrefix={arXiv},
          primaryClass={gr-qc},
          url={https://arxiv.org/abs/2501.13846},
    }

Acknowledgements
----------------

We appreciate the useful comments from Thomas Dent and Nikolaos Stergioulas.
NN acknowledges the support of Joseph Bayley, Michael Williams and Christian
Chapman-Bird.  NN is supported by the College Scholarship offered by the
School of Physics and Astronomy (2021–2025), University of Glasgow.
CM is supported by STFC grant ST/Y004256/1.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
