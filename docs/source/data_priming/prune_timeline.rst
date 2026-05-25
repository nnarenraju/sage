Cleaning the Timeline
======================

Raw GWOSC segments include very short gaps, known GW events, and lock-loss artefacts.
:meth:`~sage.data.primer.TimelineQuery.prune_segments` removes all of these in one call.

.. code-block:: python

    tq.prune_segments(
        rm_short_segments=True,
        rm_min_duration=22.0,   # drop segments shorter than 22 s
        rm_allevents=True,       # excise known GW events from GWOSC
        rm_window_length=30,     # ± 30 s around each event GPS time
    )

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``rm_short_segments``
     - If ``True``, drop segments whose duration is below ``rm_min_duration``.
   * - ``rm_min_duration``
     - Minimum segment duration in seconds. Segments shorter than this are discarded.
       22 s is the minimum needed to safely produce a 16 s analysis window with edge
       buffering.
   * - ``rm_allevents``
     - If ``True``, fetch the GWOSC event catalogue and excise a window around each
       event GPS time. This prevents known signals from contaminating the noise training
       set.
   * - ``rm_window_length``
     - Half-width (in seconds) of the excision window around each event.
       ``[event_gps − rm_window_length, event_gps + rm_window_length]`` is removed.

Why remove events?
------------------

Training on segments that contain real GW signals introduces a subtle bias: the network
sees confirmed signals at their true SNRs, which is not representative of the
signal-injection distribution used during training. Excising a ±30 s window around each
catalogued event is a conservative but safe default.

Inspecting the result
----------------------

After pruning, ``tq.timeline`` is updated in-place. You can inspect durations and
inter-segment gaps:

.. code-block:: python

    segs = tq.timeline[0]["segments"]

    # Segment durations
    durations = segs[:, 1] - segs[:, 0]

    # Gaps between consecutive segments
    gaps = segs[1:, 0] - segs[:-1, 1]
