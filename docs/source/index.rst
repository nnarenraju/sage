.. Sage documentation master file

Sage — Gravitational Wave Detection Pipeline
============================================

.. image:: /_static/sage_logo.png
   :alt: Sage logo
   :width: 380px
   :align: center

|

|DOI| |ASCL| |CI| |codecov| |Python| |PyTorch| |License|

.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20375078-blue
   :target: https://doi.org/10.5281/zenodo.20375078
   :alt: DOI

.. |ASCL| image:: https://img.shields.io/badge/ascl-4712-blue.svg?colorB=262255
   :target: https://www.ascl.net/code/v/4712
   :alt: ASCL

.. |CI| image:: https://github.com/nnarenraju/sage/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/nnarenraju/sage/actions/workflows/ci.yml
   :alt: CI

.. |codecov| image:: https://codecov.io/github/nnarenraju/sage/branch/main/graph/badge.svg?token=RLAAMEZEZ6
   :target: https://codecov.io/github/nnarenraju/sage
   :alt: codecov

.. |Python| image:: https://img.shields.io/badge/python-3.9%2B-blue
   :alt: Python

.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-2.1%2B-orange
   :alt: PyTorch

.. |License| image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://github.com/nnarenraju/sage/blob/main/LICENSE
   :alt: License: GPL v3

|

**Sage** is a complete, end-to-end machine-learning pipeline for gravitational-wave (GW)
compact binary coalescence (CBC) detection. Training operates entirely on-the-fly — no
pre-computed datasets required — with waveforms and noise generated per batch to eliminate
data-reuse biases.

Sage detects ~11.2% more signals than benchmark PyCBC matched-filtering and ~48.3% more than
the previous best-performing ML pipeline at a false alarm rate of one per month, while
remaining robust to out-of-distribution PSDs and non-Gaussian transient artefacts.

The methods are described in:

   *Identifying and Mitigating Machine Learning Biases for the Gravitational-Wave Detection
   Problem* — Nagarajan & Messenger, Phys. Rev. D **112**, 103002 (2025).
   `[paper] <https://link.aps.org/doi/10.1103/zwj9-ycyz>`__
   `[arXiv] <https://arxiv.org/abs/2501.13846>`__

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   overview

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   data_priming/index
   noise_dataset/index
   signal_dataset/index
   whitening
   multirate
   snr_estimation
   bbh_params/index
   otf_data_generation
   data_transforms
   architectures/index
   losses/index
   optimisers
   training
   validation
   full_run

.. toctree::
   :maxdepth: 1
   :caption: Reference

   citation
   autoapi/sage/index

----

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
