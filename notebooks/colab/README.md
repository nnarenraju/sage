# Sage Colab Tutorials

Interactive notebooks that let you run Sage on a free Google Colab T4 GPU with no local installation.

| # | Notebook | Description | Runtime |
|---|---|---|---|
| 1 | [Signal Generation](01_signal_generation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/01_signal_generation.ipynb) | Generate CBC waveforms with IMRPhenomD; visualise chirp mass and spin effects; compute optimal network SNR | ~8 min |
| 2 | [Data Simulation](02_data_simulation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/02_data_simulation.ipynb) | Download real O3a LIGO data; estimate PSD; whiten; inject a signal; matched-filter SNR; multirate compression | ~15 min |
| 3 | [Training & Evaluation](03_training_and_evaluation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnarenraju/sage/blob/main/notebooks/colab/03_training_and_evaluation.ipynb) | Architecture overview; train from scratch on synthetic data; evaluate with ROC curve; load pretrained checkpoint | ~25 min |

## Prerequisites

Nothing — just open a notebook in Colab and run all cells. The first cell installs Sage and its dependencies.

## Running order

The notebooks are self-contained but are designed to be read in order:
1 → 2 → 3 covers signals, noise, and learning in a logical progression.

## Notes

- A **GPU runtime** is required for waveform generation and training. In Colab: `Runtime → Change runtime type → T4 GPU`.
- Notebook 2 downloads ~8 MB of LIGO O3a data from GWOSC (~2 min on Colab's network).
- The pretrained checkpoint for Notebook 3 Part C is hosted as a GitHub Release asset. If not yet available, Part C falls back to the mini-trained model from Part B.
