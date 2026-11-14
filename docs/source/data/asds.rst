Building ASDs
=============

An amplitude spectral density (ASD) says how loud a detector's noise is at
each frequency. ``sage build-asds`` builds the ASDs a run needs from the
downloaded strain, into ``asds/`` in the data directory.

Building for a run
------------------

From inside a run directory, ``sage build-asds`` reads ``run.yaml``, lists
what the run needs, then builds it:

.. code-block:: console

   $ sage build-asds
   Building every ASD my_run reads, into /path/to/data/asds:
     1. Build the recolour bank of o3a (H1 L1)
     2. Build the recolour bank of o3b (H1 L1)
     3. Build the fiducial pooled over o3a, o3b (H1 L1)
   A part already built is checked, not built again.

As each part finishes, it says what it built, and at the end it names the
next command:

.. code-block:: console

   o3a: ASDs checked; recorded in /path/to/data/asds/o3a/asds.yaml

   o3b: ASDs checked; recorded in /path/to/data/asds/o3b/asds.yaml

   Fiducial ASDs checked: /path/to/data/asds/o3a+o3b/fiducial

   Next: sage check

Three kinds of ASD
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35
   :class: wrap

   * - Kind
     - What it is
     - What it is for
   * - Per-segment ASDs
     - The ASD of every strain piece, left empty where the piece touches bad
       time.
     - Recolouring the training noise.
   * - Recolour bank
     - The ASDs of many random 16 s stretches of strain, clear of bad time.
     - Recolouring, and making the fiducial.
   * - Fiducial
     - One ASD per detector, from the recolour banks of several observing
       runs, with spectral lines removed.
     - Whitening: every window the network sees is whitened with it.

Every run needs the fiducial, and so the recolour banks it is made from. A
run that recolours its training noise also needs the per-segment ASDs of
that noise. With ``two_detector``, which recolours its O3b training noise,
the list grows:

.. code-block:: console

   $ sage build-asds
   Building every ASD my_run reads, into /path/to/data/asds:
     1. Build the recolour bank of o3a (H1 L1)
     2. Build the per-segment ASDs and recolour bank of o3b (H1 L1)
     3. Build the fiducial pooled over o3a, o3b (H1 L1)
   A part already built is checked, not built again.

.. list-table::
   :header-rows: 1
   :widths: 45 55
   :class: wrap

   * - Folder
     - What it holds
   * - ``/path/to/data/asds/o3b/segment_psds/``
     - The per-segment ASDs of O3b.
   * - ``/path/to/data/asds/o3b/recolour_psds/``
     - The recolour bank of O3b.
   * - ``/path/to/data/asds/o3b/asds.yaml``
     - How O3b's ASDs were built, and what was built.
   * - ``/path/to/data/asds/o3a+o3b/fiducial/``
     - The fiducial, one ASD per detector.

How each is made is in :doc:`../methods/data/index`, and how they are used
in :doc:`../methods/noise/recolouring` and
:doc:`../methods/transforms/whitening`.

Stopping and resuming
---------------------

A build can stop at any time, and the same command resumes it. Parts already
built are checked rather than built again.

Building for observing runs
---------------------------

``sage build-asds`` can also build one kind of ASD for observing runs
directly, without a run directory:

.. code-block:: console

   $ sage build-asds o3b --segments
   $ sage build-asds o3a o3b --recolour-bank
   $ sage build-asds o3a o3b --fiducial

``--fiducial`` pools the observing runs it names into one fiducial, and
needs their recolour banks first. ``--detectors`` limits a build to some
detectors.

``sage build-asds --help`` shows every option:

.. code-block:: console

   $ sage build-asds --help
   usage: sage build-asds [-h] [--segments | --recolour-bank | --fiducial]
                          [--detectors DETECTORS [DETECTORS ...]] [--workers WORKERS]
                          [targets ...]

   Build the per-segment ASDs, recolour bank or fiducial from downloaded strain.

   positional arguments:
     targets               observing runs (e.g. o3b; the runs to pool with --fiducial) with
                           --segments, --recolour-bank or --fiducial, or a run directory or a
                           directory inside it (default: here)

   options:
     -h, --help            show this help message and exit
     --segments            per-segment ASDs of each observing run
     --recolour-bank       recolour bank of each observing run
     --fiducial            fiducial ASDs pooled over the observing runs
     --detectors DETECTORS [DETECTORS ...]
                           with observing runs: these detectors only (default: all GWOSC lists)
     --workers WORKERS     processes for the fiducial's line tests (default: 16)

If the strain is missing
------------------------

``sage build-asds`` lists the missing files and points to
``sage download``:

.. code-block:: console

   $ sage build-asds
   Building every ASD my_run reads, into /path/to/data/asds:
     1. Build the recolour bank of o3a (H1 L1)
     2. Build the recolour bank of o3b (H1 L1)
     3. Build the fiducial pooled over o3a, o3b (H1 L1)
   A part already built is checked, not built again.

   error: strain files are missing:
     /path/to/data/o3a/training/data_H1_O3a.bin
     /path/to/data/o3a/training/data_H1_O3a_segments.json
     ...
   fix: sage download
