Getting Segments
================

:mod:`sage.data.primer` provides :class:`~sage.data.primer.TimelineQuery` to query
GWOSC for science-mode segments across any combination of detectors and observing runs.

Listing available detectors and runs
-------------------------------------

.. code-block:: python

    from sage.data.primer import TimelineQuery, get_all_detnames, get_all_runnames

    print(get_all_detnames())
    # {'H1', 'L1', 'V1', 'G1', 'K1', 'H2'}

    print(get_all_runnames())
    # ['O1', 'O2', 'O3GK', 'O3a', 'O3b', 'O4a', 'O4b1Disc', ...]

Creating a timeline query
--------------------------

Pass a list of detectors and observing runs. Setting ``auto_clean_empty_timelines=True``
silently drops any detector/run combination that returns no segments.

.. code-block:: python

    tq = TimelineQuery(
        detector=["H1", "L1", "V1"],
        observing_run=["O3b"],
        auto_clean_empty_timelines=True,
    )

    tq.download_segments()

After ``download_segments()`` returns, ``tq.timeline`` holds a structured NumPy array
with one row per (detector, flag) pair. Each row contains:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``detector``
     - Detector name, e.g. ``'H1'``
   * - ``flag``
     - DQ flag used for the query, e.g. ``'H1_DATA'``
   * - ``start_time``
     - GPS start of the run epoch
   * - ``end_time``
     - GPS end of the run epoch
   * - ``observing_run``
     - Observing run name
   * - ``segments``
     - ``(N, 2)`` float64 array of ``[gps_start, gps_end]`` pairs

.. code-block:: python

    # Inspect the first detector's segment array
    print(tq.timeline[0]["detector"])   # 'H1'
    print(tq.timeline[0]["segments"])   # array([[1256663440., ...], ...])
