Reviewer Guide
===============

.. warning::

   This guide is under construction. The content below outlines what will be available here
   once the guide is complete.

----

The Reviewer Guide is aimed at referees and readers who want to reproduce, verify, or
challenge the results presented in the Sage paper. The goal is to make independent
verification as frictionless as possible.

----

What Will Be Here
------------------

Automated Review Tools
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   A dedicated reviewer tool suite is in preparation. It will allow referees to reproduce
   key paper results with a single command — no manual configuration required.

The planned tools will include:

- **Result reproducer** — a CLI that reruns the full MLGWSC-1 evaluation from a published
  checkpoint and prints the sensitive distance and FAR curves used in the paper
- **Bias audit tool** — runs each of the 11 bias mitigation toggles independently so
  referees can verify that each mitigation has the claimed effect on detection performance
- **Waveform validator** — compares Sage IMRPhenomD/Pv2 outputs against LALSuite at
  a set of standard test points and reports mismatches above a threshold
- **Checkpoint inspector** — loads a checkpoint and prints a human-readable summary of
  the model architecture, training config, and training history

Reproducing the Paper Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step-by-step instructions for reproducing each table and figure in the paper from
publicly available data and checkpoints. Each step will list:

- Input data (GWOSC segment, injection set, or synthetic)
- The exact command to run
- Expected output and how to interpret it
- Approximate runtime on a reference GPU

Data Availability
~~~~~~~~~~~~~~~~~~

- Where to download the O3b noise data used in training (GWOSC)
- Where to download the MLGWSC-1 injection set
- Download instructions for published model checkpoints

Common Reviewer Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~

A curated list of questions that arose during peer review of the paper, with detailed
technical answers — covering methodology choices, baseline comparisons, and robustness
claims.

Getting Help
~~~~~~~~~~~~~

If a result cannot be reproduced or a claim seems inconsistent, please open a
`GitHub issue <https://github.com/nnarenraju/sage/issues>`__ with the label
``reproducibility``. We treat reproducibility issues as high-priority bugs.
