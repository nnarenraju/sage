.. Sage documentation master file

Sage — Gravitational Wave Detection Pipeline
============================================

.. image:: https://zenodo.org/badge/482025216.svg
   :target: https://doi.org/10.5281/zenodo.17290133
   :alt: DOI

**Sage** is a machine-learning based binary black-hole search pipeline
designed to identify and mitigate supervised-learning biases that hinder
generalisation and detection performance.

Reference paper:
`Identifying and Mitigating Machine Learning Biases for the Gravitational-Wave Detection Problem
<https://arxiv.org/abs/2501.13846>`_ — Nagarajan & Messenger (2025).

Key results on the MLGWSC-1 benchmark (O3a noise):

* ~11.2% more detections than PyCBC at 1 false alarm per month
* ~48.3% more detections than the previous best ML pipeline

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   autoapi/sage/index


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
2. Real LIGO O3b noise windows are asynchronously fetched from a memory-mapped
   file (:class:`~sage.data.noise.real_noise.MemmapNoiseSampler`).
3. Signals are injected at random positions in the noise batch.
4. The pipeline applies FD whitening
   (:class:`~sage.dsp.whiten.FiducialWhitening`) and dyadic multirate
   sampling (:class:`~sage.dsp.multirate_sampling.MultirateSampler`).
5. The loss is computed using
   :class:`~sage.architecture.custom_losses.loss_functions.BCEWithFARLoss`
   — a composite of BCE, heteroscedastic regression, coupling regularisation,
   partial-AUC maximisation, and focal loss.

Hard-sample mining (:class:`~sage.data.noise.hard_mining.HardSampleMiner`)
and glitch oversampling
(:class:`~sage.data.noise.glitch_sampler.GlitchOversampledNoiseSampler`)
are used to address dataset imbalances and adversarial noise.

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
