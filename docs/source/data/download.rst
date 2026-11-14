Downloading Strain
==================

``sage download`` fetches detector strain from GWOSC, the Gravitational Wave
Open Science Center, into the data directory.

Downloading for a run
---------------------

From inside a run directory, ``sage download`` reads ``run.yaml`` and
fetches the strain the run trains and validates on. For ``quick``, that is
O3a and O3b, for H1 and L1:

.. code-block:: console

   $ sage download
   Downloading the training strain of o3a (H1 L1), o3b (H1 L1) into /path/to/data. For each run and detector:
     1. Download the strain from GWOSC in 512 s pieces, cutting out 30 s around every GWOSC event; failed pieces are retried
     2. Check the strain against its segment list
     3. Fetch the bad time from GWOSC: CBC Category 2 and hardware injections
     4. Find the spans of exactly-zero strain
   Then write each run's record. A stopped download resumes.

   o3a H1
   [1/4] Download the strain from GWOSC in 512 s pieces, cutting out 30 s around every GWOSC event; failed pieces are retried

It works through each observing run and detector in turn, one step at a
time. When an observing run is done, it prints what it downloaded:

.. code-block:: console

   o3b H1: 20286 pieces, 120.15 days, GPS 1256663443-1269363618
   o3b L1: 19941 pieces, 118.08 days, GPS 1256655666-1269363615
   Recorded in /path/to/data/o3b/training/dataset.yaml

and at the end, the next command:

.. code-block:: console

   Next: sage build-asds

What it downloads
-----------------

The strain is downloaded in pieces of 512 s, with 30 s cut out around every
event GWOSC lists, so training sees no known signals. It is resampled to
2048 Hz, high-passed at 15 Hz and stored as 32-bit floats.

With the strain come two kinds of time that training leaves out:

* **Bad time**, from GWOSC: CBC Category 2 data-quality flags and hardware
  injections.
* **Zero spans**: stretches where the strain is exactly zero.

Each observing run gets a folder, here O3b's:

.. list-table::
   :header-rows: 1
   :widths: 45 55
   :class: wrap

   * - ``/path/to/data/o3b/training/``
     - What it holds
   * - ``data_H1_O3b.bin``
     - The strain pieces, one after another.
   * - ``data_H1_O3b_segments.json``
     - Where each piece starts and ends, in GPS time and in the file.
   * - ``data_H1_O3b_bad_time.json``
     - The bad time.
   * - ``data_H1_O3b_zero_runs.json``
     - The zero spans.
   * - ``data_H1_O3b_failed_segments.json``
     - Pieces that could not be downloaded.
   * - ``dataset.yaml``
     - How the strain was made, and what was downloaded.

and the same files for every other detector.

Stopping and resuming
---------------------

A download can stop at any time, and the same command resumes it. Parts
already finished are checked rather than downloaded again. Pieces that
failed to download are listed in the ``_failed_segments.json`` file, and the
next ``sage download`` retries them.

Downloading an observing run
----------------------------

``sage download`` can also fetch an observing run's strain directly, without
a run directory. ``--train`` downloads it as training strain, for every
detector GWOSC lists:

.. code-block:: console

   $ sage download o3b --train

and ``--detectors`` limits it to some:

.. code-block:: console

   $ sage download o3b --train --detectors H1 L1

``--search`` downloads it for searching instead, as :doc:`../searching`
explains.

``sage download --help`` shows every option:

.. code-block:: console

   $ sage download --help
   usage: sage download [-h] [--train | --search] [--detectors DETECTORS [DETECTORS ...]]
                        [--workers WORKERS]
                        [target]

   Download the data an observing run or a run directory needs.

   positional arguments:
     target                observing run (e.g. o3b) with --train or --search, or a run directory or a
                           directory inside it (default: here)

   options:
     -h, --help            show this help message and exit
     --train               with an observing run: training strain (512 s pieces, events cut out), its
                           bad time and zero spans
     --search              with an observing run: search strain (whole GWOSC segments, nothing
                           removed), its bad time and zero spans
     --detectors DETECTORS [DETECTORS ...]
                           with an observing run: these detectors only (default: all GWOSC lists)
     --workers WORKERS     parallel downloads and fiducial line tests (default: 16)

``--workers`` sets how many pieces download at once.

If GWOSC cannot be reached
--------------------------

``sage download`` stops, and says to run it again once GWOSC is reachable:

.. code-block:: console

   $ sage download
   error: GWOSC could not be reached from here: HTTPSConnectionPool(host='gwosc.org', port=443): Read timed out. (read timeout=10)
   fix: once https://gwosc.org is reachable: sage download .
