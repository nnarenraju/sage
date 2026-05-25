Splitting into Mini-Segments
=============================

The downloader fetches one mini-segment at a time in parallel. Splitting the timeline
into fixed-length chunks beforehand lets you parallelise downloads trivially and ensures
every stored segment has a known, uniform duration.

Buffer calculation
------------------

Whitening with inverse-spectrum truncation introduces edge artefacts. Sage trims a small
buffer from each end of every downloaded segment. The buffer length must be an exact
integer number of samples at the target sample rate, so it is rounded up:

.. code-block:: python

    import math

    # data_cfg.sample_rate is e.g. 2048 Hz
    buffer = math.ceil(0.2 * data_cfg.sample_rate) / data_cfg.sample_rate
    # buffer = 0.200195... seconds  (410 samples at 2048 Hz)

Splitting
---------

.. code-block:: python

    tq.split_into_mini_segments(
        mini_segment_length=512.0 + (buffer * 2.0),  # ~512.4 s per chunk
        minimum_segment_duration=16.0,                # discard trailing runt chunks
    )

Each original segment is subdivided into non-overlapping chunks of
``mini_segment_length`` seconds. Any trailing piece shorter than
``minimum_segment_duration`` is discarded.

After splitting, ``tq.timeline[i]["segments"]`` can contain tens of thousands of rows:

.. code-block:: text

    H1  →  20 286 mini-segments
    L1  →  19 941 mini-segments
    V1  →  19 859 mini-segments

Sanity check
------------

:meth:`~sage.data.primer.TimelineQuery.sanity_check_mini_segments` verifies that every
mini-segment has the expected length and that no chunk is shorter than the minimum:

.. code-block:: python

    tq.sanity_check_mini_segments(
        mini_segment_length=512.0 + (buffer * 2.0),
        minimum_segment_duration=16.0,
        verbose=True,
    )

Run this before downloading to catch any edge cases early.
