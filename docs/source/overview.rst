Overview
========

Sage is an end-to-end supervised learning pipeline that tackles gravitational-wave CBC
detection by systematically identifying and mitigating 11 interconnected learning biases.
All modules are optimised for CPU and GPU (``torch.compile``-compatible) usage.

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

Architecture
------------

Sage uses a **two-stage neural network**:

1. **Frontend** — per-detector multi-scale 1D CNN
   (:class:`~sage.architecture.frontend.mscnn1d.ConvBlock`) that extracts
   temporal features across a wide range of time scales using five parallel
   convolutions at different kernel sizes.

2. **Backend** — 2D ResNet with CBAM (Convolutional Block Attention Module)
   attention (:class:`~sage.architecture.backend.resnet2d_cbam.ResNet`) that
   processes the stacked per-detector feature maps and produces a compact
   feature vector.

The network head outputs:

* A **ranking statistic** (raw logit for BCE classification).
* **Heteroscedastic point estimates** (mean + log-variance) for each
  regression target (e.g. chirp mass, coalescence time).

Training pipeline
-----------------

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


Waveform generation
-------------------

Two GPU-native batched approximants are provided:

* :class:`~sage.data.waveform.approximants.IMRPhenomD.IMRPhenomD` — aligned-spin
  frequency-domain waveforms, ``torch.compile``-compatible (``GRAPH_READY = True``).
* :class:`~sage.data.waveform.approximants.IMRPhenomPv2.IMRPhenomPv2` — precessing-spin
  extension via the PhenomPv2 "twist-up" formalism.

Waveform parameters are drawn from a YAML-configured prior using
:class:`~sage.data.waveform.sampler.DistributionSampler`.

Signal processing
-----------------

The DSP stack (:mod:`sage.dsp`) includes:

* **Whitening** — frequency-domain whitening with inverse spectrum truncation.
* **Multirate sampling** — dyadic time-domain downsampling to focus resolution
  around the merger.
* **FD multibanding** — frequency-domain multirate representation for
  compute-efficient signal coverage.
* **Prior-median heterodyning** — carrier-frequency removal that collapses
  the CBC chirp to a narrow band, reducing required sample rates.

Noise handling
--------------

* **Real strain noise** — GWOSC O3a/O3b data accessed via memory-mapped files.
* **Coloured / recoloured noise** — synthetic noise coloured to a target PSD.
* **Glitch oversampling** — GW Classify-labelled glitch segments are oversampled
  to expose the network to realistic transient artefacts.
