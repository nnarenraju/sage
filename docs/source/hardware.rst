Hardware & Compute Guide
=========================

.. warning::

   This page is under construction. Specific numbers and benchmarks will be added once
   hardware configurations are finalised.

----

Minimum Requirements
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Requirement
   * - GPU
     - *TBA* — NVIDIA GPU with CUDA support required
   * - GPU VRAM
     - *TBA*
   * - System RAM
     - *TBA*
   * - Storage
     - *TBA* — noise data is large; fast local SSD strongly recommended
   * - CUDA version
     - *TBA*
   * - PyTorch version
     - ≥ 2.1 (for ``torch.compile`` support)

----

Recommended Configurations
----------------------------

Development (small-scale experiments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Specification
   * - GPU
     - *TBA*
   * - VRAM
     - *TBA*
   * - RAM
     - *TBA*
   * - Storage
     - *TBA*

Production (full O3b run)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Specification
   * - GPU
     - *TBA*
   * - VRAM
     - *TBA*
   * - RAM
     - *TBA*
   * - Storage
     - *TBA*

----

Throughput Estimates
---------------------

.. note::

   All throughput figures will be filled in after systematic benchmarking.
   Numbers depend heavily on waveform model, batch size, and ``torch.compile`` mode.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Configuration
     - Batches / second
     - GPU utilisation
   * - IMRPhenomD, batch 32
     - *TBA*
     - *TBA*
   * - IMRPhenomPv2, batch 32
     - *TBA*
     - *TBA*
   * - IMRPhenomPv2 + ``torch.compile``
     - *TBA*
     - *TBA*

----

Storage Layout
---------------

Observing Run data download
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   data/
   └── O3/
       ├── H1/         # Hanford strain (.bin files, one per segment)
       ├── L1/         # Livingston strain
       ├── V1/         # Virgo strain
       └── psds/       # Pre-computed noise PSDs

Estimated total size: *TBA* (Observing runs are 100s of GB at 2048 Hz sampling rate — download selectively if storage is limited).

Checkpoint storage
~~~~~~~~~~~~~~~~~~~

Each checkpoint saves model weights, optimiser state, and GradScaler state.
Approximate size per checkpoint: *TBA*.

----

Cluster Setup
--------------

SLURM example
~~~~~~~~~~~~~~

A minimal SLURM batch script will be provided here once the recommended GPU allocation
and time limits are benchmarked.

.. code-block:: bash

   # Placeholder — numbers TBA
   #SBATCH --gres=gpu:TBA
   #SBATCH --mem=TBAG
   #SBATCH --time=TBA
   python runs/o3b/train.py

Environment modules
~~~~~~~~~~~~~~~~~~~~

Specific module load commands for common HPC clusters (e.g., CUDA, conda activation)
will be listed here once tested.

----

Tips for Low-VRAM Systems
--------------------------

If your GPU has limited VRAM, try the following in order:

1. Halve ``batch_size`` in ``RunCFG`` — roughly halves peak activation memory.
2. Enable ``autocast = True`` in ``RunCFG`` to use float16 during the forward pass.
3. Use ResNet-50 (``backend_resnet_size=50``) instead of larger backends.
4. Reduce ``prefetch`` in ``MemmapNoiseSampler``.

See also :doc:`troubleshooting` for GPU out-of-memory fixes.
