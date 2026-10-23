# Changelog

All notable changes to Sage are documented here.

## [Unreleased]

## [0.1.0] - 2026-05-25

This release is a complete rewrite and restructuring of the Sage codebase (PR #1). The
previous monolithic scripts accompanying the PRD paper have been replaced by a fully
modular `sage/` Python package with clean separation of concerns across DSP, data
handling, architecture, training, and utilities.

### Added — package structure
- `sage.core` — configuration registry, logging, constants, type utilities, seeding,
  interpolation, and shared math helpers.
- `sage.dsp` — FFT utilities, whitening, inverse spectrum truncation, Welch PSD
  estimation, FIR filters, time-domain multirate sampling, frequency-domain
  multibanding, and prior-median heterodyning.
- `sage.data.noise` — real noise memory-mapping (`real_noise`), coloured and white noise
  generators, recolouring, glitch noise and glitch sampler, and hard noise mining.
- `sage.data.primer` — automated download and preparation of GWOSC data releases,
  observing-run segments, PSDs, MLGWSC data, and pre-trained checkpoints.
- `sage.data.asd` — ASD estimation (Welch + sqrt), ASD reading, and blackout utilities.
- `sage.data.waveform` — parameter sampling distributions, IMRPhenomD/PhenomPv2
  waveform generation, multi-detector projection, SNR calculation, and tapering.
- `sage.architecture` — modular frontend (MSCNN1D, MSCNN1D-CBAM variants), backend
  (ResNet2D-CBAM, ResNet3D-CBAM), attention zoo (cross-attention), custom losses
  (heteroscedastic regression), and an architecture manager.
- `sage.factory` — training and validation loops, learning-rate schedulers, callbacks,
  and checkpoint management.
- `sage.benchmark.mlgwsc1` — MLGWSC-1 evaluation harness for standardised pipeline
  benchmarking.
- `sage.plotting` — diagnostic plots for ranking statistics, efficiency curves, ROC
  curves, and parameter studies.
- `sage.presets` — pre-built configuration presets for common training modes.
- `sage.utils` — checkpointing helpers, timing utilities, and Condor job submission.

### Added — DSP
- `sage.dsp.multibanding` — frequency-domain multibanding compressor
  (`FrequencyMultibandCompressor`, `FrequencyBandLayout`, `FrequencyBand`). GPU and
  `torch.compile` compatible. Supports `sample` and `mean` pooling modes. Includes
  `make_dyadic_frequency_bands`, `make_prior_informed_frequency_bands`, and
  `make_empirical_frequency_bands` band constructors.
- `sage.dsp.heterodyning` — prior-median frequency-domain heterodyning
  (`apply_heterodyne`, `compute_reference_phase`, `make_median_reference_binary`,
  `residual_chirp_time`).

### Added — data and noise mining
- Hard noise mining (`sage.data.noise.hard_mining`): `HardSampleBuffer` and
  `HardSampleMiner` for periodic mining of difficult background windows during training.
- Low-FAR noise mining (`sage.data.noise.lowfar_noise`): `BruteForceMiner`,
  `MAPElitesMiner` (quality-diversity archive over GPS time), `CEMRareEventMiner`
  (cross-entropy method with diversity floor), and `StartTimeDataset` for offline mining
  and replay.

### Added — usability and documentation
- Google Colab tutorials (no local install required): signal generation with IMRPhenomD,
  realistic data simulation and whitening, training and evaluating a GW detector.
- ELI5 introductory notebooks explaining core concepts for new users.
- Full API documentation at [sage-gw.readthedocs.io](https://sage-gw.readthedocs.io).
- `CONTRIBUTING.md` with PR workflow and style notes.
- `CITATION.cff` for software citation metadata.
- ASCL record: [ascl:4712](https://www.ascl.net/code/v/4712).
- CI workflow (GitHub Actions) with pytest and Codecov coverage reporting.
- Comprehensive test suite under `tests/`.
- Verification notebooks: `notebooks/multibanding.ipynb` (decimation cross-check
  and mismatch study across 200 prior waveforms, max mismatch 2×10⁻⁵) and
  `notebooks/heterodyning.ipynb` (reference binary selection, residual chirp-time
  envelope, compression comparison).

### Fixed
- Critical bug: `hc` polarisation calculation was incorrect because `hp` was mutated
  before `final_hc` was computed.
- `BatchToFrequencyDomain` FFT normalisation was missing, causing incorrect spectral
  amplitudes.
- Recursion error when saving nested configs to disk.
- Parameter constraints and transforms were not applied to standardiser samples.
- Scaler state was not included in checkpoint saves.
- Binary saving scheme broken for `num_workers=1`.

## [0.0.1] - 2025-11-01

### Added
- Initial public release accompanying the Phys. Rev. D paper (DOI: 10.1103/zwj9-ycyz).
- End-to-end CBC search pipeline: waveform generation, detector projection, noise
  handling, whitening, neural-network training and validation.
- Time-domain multirate sampling (`sage.dsp.multirate_sampling`).
- Neural network architectures: frontend, backend, attention modules.
- Reproducibility notebooks and run scripts under `repro/` and `runs/`.
- Diagnostic plotting utilities.
