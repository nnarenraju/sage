Checking a Run
==============

``sage check`` makes every check ``sage train`` makes when it starts, then
takes one real training step. A mistake in the settings or the data shows in
a minute or two, instead of after the first epoch.

The check is small: a batch of two windows and few read workers. It runs on
the GPU if one is visible, and on the CPU if not, so it also works on a
machine without a GPU. It writes nothing but a flag, ``flags/checked``.

For ``quick``, on a machine without a GPU:

.. code-block:: console

   $ sage check
   Checking on the CPU (no GPU is visible), uncompiled, with a batch of 2 and 2 read workers:
     1. Check the settings and prior, and a started run's record in resolved/
     2. Check the data files
     3. Build the samplers, preprocessing and network
     4. Compute the derived values; a started run's match resolved/derived.yaml
     5. Take one training step
     6. Build the validation streams

   [1/6] Check the settings and prior, and a started run's record in resolved/
     done in 0.0 s

   [2/6] Check the data files
     done in 0.0 s

   [3/6] Build the samplers, preprocessing and network
   00:03:23 | INFO    | Run my_run from epoch 0, seed 150914
   00:03:24 | INFO    | noise o3b H1: 20286 segments, 120.15 days, GPS 1256663443.2001953-1269350583.8566895
   00:03:24 | INFO    | noise o3b L1: 19941 segments, 118.08 days, GPS 1256657155.4020996-1269342291.244629
     done in 4.2 s

   [4/6] Compute the derived values; a started run's match resolved/derived.yaml
     done in 0.0 s

   [5/6] Take one training step
   00:03:28 | INFO    | Epoch 0 | training: 1 iterations, batch 2
   00:03:32 | INFO    |   epoch 0 | 100% (1/1) | loss 0.6944 | 0.2 it/s | eta 0s
   00:03:32 | INFO    | Epoch 0 | train loss 0.69442 | components [0.69276, 0.11145, -0.10830, 0.32794] | lr 2.000e-04 | 0.2 it/s | 4s
     done in 4.4 s

   [6/6] Build the validation streams
   00:03:32 | INFO    | noise o3a H1: 22874 segments, 135.42 days, GPS 1238235751.2001953-1253962208.2890625
   00:03:32 | INFO    | noise o3a L1: 24378 segments, 144.24 days, GPS 1238166018.2001953-1253970235.4274902
     done in 0.5 s

   All checks passed; one training step on the CPU: loss 0.69442. Wrote flags/checked

   Next: sage train

Checking is optional: ``sage train`` makes the same checks itself before it
trains.

The checked flag
----------------

``sage status`` shows that the run has been checked:

.. code-block:: console

   $ sage status
   my_run: training has not started
   Flags: checked (since 2026-11-14 23:03:32 UTC)
   Next: sage train

If ``run.yaml`` or ``prior.yaml`` changes after the check, ``sage status``
says what changed, and the run is no longer shown as checked:

.. code-block:: console

   $ sage status
   my_run: training has not started
   Changed since sage check (train.epochs: 6 -> 8); to check: sage check
   Next: sage train

When a check fails
------------------

``sage check`` stops at the first problem and says how to fix it. Without
the downloaded strain, for example:

.. code-block:: console

   $ sage check
   error: downloaded files this run reads are missing:
     /path/to/data/o3b/training/dataset.yaml
     /path/to/data/o3b/training/data_H1_O3b.bin
     /path/to/data/o3b/training/data_H1_O3b_segments.json
     ...
   fix: sage download

Mistakes in ``run.yaml`` and ``prior.yaml`` are on
:doc:`../runs/mistakes`.

``sage check --help`` shows every option:

.. code-block:: console

   $ sage check --help
   usage: sage check [-h] [directory]

   Run every start-up check and one real training step; marks the run checked.

   positional arguments:
     directory   run directory, or a directory inside it (default: here)

   options:
     -h, --help  show this help message and exit
