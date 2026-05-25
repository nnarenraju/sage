.. Sage documentation master file

Physics-informed Representation Learning for Gravitational-wave Discovery
=========================================================================

.. image:: /_static/rectangular_logo.png
   :alt: Sage logo
   :width: 480px
   :align: center

|

An end-to-end PyTorch pipeline for compact binary coalescence detection —
from raw gravitational-wave strain to a bespoke neural network, with systematic bias 
mitigations built in. 

.. admonition:: Collaborate with us!

   We welcome collaborations — reach out if you want a production-level GW search built for your data!
   Visit the :doc:`collaboration` page to get in touch.

.. warning::

   These docs are actively being built and are subject to change. Individual sections
   may contain incomplete explanations, bugs, or mistakes. Additional plots and figures
   will be added throughout to make each section easier to follow.

----

.. grid:: 1 2 2 4
   :gutter: 3

   .. grid-item-card:: :octicon:`download` Install
      :link: installation
      :link-type: doc

      Set up Sage via conda or pip. GPU support included out of the box.

      +++
      :doc:`Installation guide <installation>`

   .. grid-item-card:: :octicon:`book` User Guides
      :link: data_priming/index
      :link-type: doc

      Hands-on walkthroughs: data download, waveforms, DSP, and training.

      +++
      :doc:`Browse guides <data_priming/index>`

   .. grid-item-card:: :octicon:`code-square` API Reference
      :link: autoapi/sage/index
      :link-type: doc

      Autoapi documentation for every module, class, and function in Sage.

      +++
      :doc:`Browse API <autoapi/sage/index>`

   .. grid-item-card:: :octicon:`rocket` Colab Tutorials
      :link: quickstart
      :link-type: doc

      Zero-install notebooks — run the full pipeline in your browser.

      +++
      :doc:`Open notebooks <quickstart>`

----

.. admonition:: State-of-the-art performance on MLGWSC-1

   At a false alarm rate of one per month, Sage detects **+11.2%** more signals
   than the PyCBC matched-filter benchmark and **+48.3%** more signals than the
   previous best ML pipeline — with demonstrated robustness to out-of-distribution
   noise PSDs and non-Gaussian glitches.

----

**Sage** is a complete, end-to-end machine-learning pipeline for searching gravitational-wave
(GW) detector data for compact binary coalescence (CBC) signals. The package spans the entire
research workflow: automated download and preparation of GWOSC data releases and PSDs;
realistic noise simulation (real strain, coloured, recoloured, and glitch-injected); waveform
generation and multi-detector projection via IMRPhenomD and IMRPhenomPv2; signal processing
including whitening, inverse spectrum truncation, time-domain multirate sampling,
frequency-domain multibanding, and prior-median heterodyning; neural network training;
and diagnostic evaluation and benchmarking against previous results.

All data-generation, signal-processing, and neural-network components are written in PyTorch
and are fully ``torch.compile``-compatible, enabling significant GPU throughput improvements
without any code changes. Training operates entirely on-the-fly — no pre-computed datasets are
required — with waveforms and noise windows generated per batch to eliminate data-reuse biases.

Sage systematically identifies and mitigates 11 interconnected supervised-learning biases that
degrade detection performance and generalisation. On the Machine Learning Gravitational-Wave
Search Challenge injection study, Sage detects approximately 11.2% more signals than the
benchmark PyCBC matched-filter analysis and approximately 48.3% more signals than the previous
best-performing ML pipeline at a false alarm rate of one per month, while demonstrating
robustness to out-of-distribution noise PSDs and non-Gaussian transient artefacts.

The modular design — with interchangeable frontends, backends, attention mechanisms, and
configurable presets — makes Sage straightforward to adapt for new architectures, parameter
spaces, or observing runs. Google Colab tutorials allow zero-installation experimentation.

The methods are described in:

   *Identifying and Mitigating Machine Learning Biases for the Gravitational-Wave Detection
   Problem* — Nagarajan & Messenger, Phys. Rev. D **112**, 103002 (2025).
   `[paper] <https://link.aps.org/doi/10.1103/zwj9-ycyz>`__
   `[arXiv] <https://arxiv.org/abs/2501.13846>`__

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   installation
   quickstart
   collaboration

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   overview
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
   :caption: Developer Guide
   :hidden:

   developer_guide

.. toctree::
   :maxdepth: 1
   :caption: Reviewer Guide
   :hidden:

   reviewer_guide

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   benchmarks
   citation
   autoapi/sage/index

----

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
