Quick Start
===========

Getting started
----------------

The :doc:`User Guide <overview>` covers the full pipeline from data download to training
and evaluation. For complete, working examples of how Sage is configured and run in
practice, see the scripts under
`runs/ <https://github.com/nnarenraju/sage/tree/main/runs>`_.

Colab tutorials
---------------

No local install needed — run Sage on a free Google Colab T4 GPU:

.. list-table::
   :header-rows: 1
   :widths: 5 60 35

   * - #
     - Notebook
     - Launch
   * - 1
     - Signal generation with IMRPhenomD
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/01_signal_generation.ipynb
          :alt: Open In Colab
   * - 2
     - Realistic data simulation and whitening
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/02_data_simulation.ipynb
          :alt: Open In Colab
   * - 3
     - Training and evaluating a GW detector
     - .. image:: https://colab.research.google.com/assets/colab-badge.svg
          :target: https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/03_training_and_evaluation.ipynb
          :alt: Open In Colab

Repository layout
-----------------

.. code-block:: text

    sage/
    ├── sage/
    │   ├── architecture/       # Frontend, backend, attention, and full networks
    │   ├── benchmark/          # Benchmark integrations and comparison utilities
    │   ├── core/               # Config, logging, constants, interpolation
    │   ├── data/
    │   │   ├── noise/          # Real noise samplers and glitch handling
    │   │   ├── primer/         # Data download and preparation utilities
    │   │   ├── psd/            # PSD generation and loading
    │   │   └── waveform/       # Parameter sampling, waveforms, projection, SNR
    │   ├── dsp/                # FFT, whitening, PSDs, multirate, multibanding
    │   ├── exec/               # Pipeline orchestration
    │   ├── factory/            # Training, validation, schedulers, callbacks
    │   ├── plotting/           # Diagnostic and publication plotting
    │   ├── presets/            # Legacy configs and shared data configs
    │   └── utils/              # Checkpointing, timing, Condor utilities
    ├── runs/                   # Run scripts for specific experiments
    ├── repro/                  # Reproducibility notebooks and configuration
    ├── notebooks/              # Exploratory notebooks
    ├── tests/                  # Lightweight tests and smoke checks
    └── docs/                   # Sphinx/ReadTheDocs documentation source
