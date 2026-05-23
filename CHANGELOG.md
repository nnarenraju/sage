# Changelog

All notable changes to Sage are documented here.

## [Unreleased]

### Added
- `sage.dsp.multibanding` — frequency-domain multibanding compressor (`FrequencyMultibandCompressor`, `FrequencyBandLayout`, `FrequencyBand`). GPU and `torch.compile` compatible. Supports `sample` and `mean` pooling modes. Includes `make_dyadic_frequency_bands`, `make_prior_informed_frequency_bands`, and `make_empirical_frequency_bands` band constructors.
- `sage.dsp.heterodyning` — prior-median frequency-domain heterodyning (`apply_heterodyne`, `compute_reference_phase`, `make_median_reference_binary`, `residual_chirp_time`).
- `notebooks/multibanding.ipynb` — verification notebook for FD multibanding: DINGO comparison, resolution criterion checks, mismatch study across 200 prior waveforms (max mismatch 2×10⁻⁵).
- `notebooks/heterodyning.ipynb` — exploration notebook for prior-median heterodyning: reference binary selection, residual chirp-time envelope, compression comparison.

## [Previous Release]

### Added
- Initial public release accompanying the Phys. Rev. D paper (DOI: 10.1103/zwj9-ycyz).
- End-to-end CBC search pipeline: waveform generation, detector projection, noise handling, whitening, neural-network training and validation.
- Time-domain multirate sampling (`sage.dsp.multirate_sampling`).
- Neural network architectures: frontend, backend, attention modules.
- Reproducibility notebooks and run scripts under `repro/` and `runs/`.
- Diagnostic plotting utilities.
