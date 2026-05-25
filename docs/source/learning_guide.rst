Learning Guide
===============

.. warning::

   This guide is under construction. The content below outlines what will be available here
   once the guide is complete.

----

The Learning Guide is designed for three audiences: complete beginners with no prior
background; physicists who want to understand the ML side of Sage; and machine-learning
practitioners who want to understand the GW physics. See :doc:`beginners` if you are
starting from scratch.

----

What Will Be Here
------------------

GW Physics Primer (for ML readers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A self-contained introduction to the gravitational-wave detection problem — no prior GW
knowledge assumed. Topics will include:

- What compact binary coalescence signals look like in time and frequency
- Why detector noise is coloured and what whitening achieves
- How matched filtering works and why it is the optimal linear detector
- What false alarm rate means and why it is the correct figure of merit
- A plain-language explanation of the MLGWSC-1 benchmark setup

ML Primer (for GW readers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A concise introduction to the supervised-learning concepts used in Sage — aimed at
physicists already comfortable with GW data analysis. Topics will include:

- What a convolutional neural network learns from time-series data
- Why on-the-fly data generation matters (and what data-reuse bias looks like in practice)
- How binary cross-entropy connects to detection statistics
- What ``torch.compile`` does and why it is safe to treat as a black box

Step-by-Step Tutorials
~~~~~~~~~~~~~~~~~~~~~~~~

Notebook-style tutorials that build from first principles to a full training run:

1. Downloading and inspecting O3b noise data
2. Generating your first IMRPhenomPv2 waveform
3. Running the whitening and multirate pipeline on a single sample
4. Training a minimal ResNet on synthetic CBC data
5. Evaluating a checkpoint against the MLGWSC-1 injection set

All tutorials will be available as Google Colab notebooks for zero-installation use.

Worked Examples
~~~~~~~~~~~~~~~~

Case studies showing how to adapt Sage for specific research questions, including:

- Changing the mass prior to target a different population
- Swapping the frontend for a different time-frequency representation
- Evaluating on a custom noise dataset

Exercises
~~~~~~~~~~

Short, self-contained exercises to test understanding — with solutions provided.

.. toctree::
   :hidden:

   beginners
