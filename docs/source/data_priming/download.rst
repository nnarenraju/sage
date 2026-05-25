Downloading from GWOSC
=======================

:class:`~sage.data.primer.DataReleaseDownloader` fetches each mini-segment from GWOSC,
resamples it, trims the corruption buffer, and writes all segments into a single
monolithic binary file per detector.

.. code-block:: python

    from sage.data.primer import DataReleaseDownloader

    drd = DataReleaseDownloader(
        segments_metadata=tq.timeline,
        save_parent_dir="/path/to/storage",
        noise_low_freq_cutoff=15.0,
        minimum_segment_duration=22.0,
        corrupt_trim_length=buffer,
        max_download_retries=15,
        retry_delay=0.5,
        num_workers=64,
        make_monolithic_file=True,
        sample_rate=data_cfg.sample_rate,
        save_bin=True,
    )

    drd.download()

Key parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Description
   * - ``segments_metadata``
     - ``tq.timeline`` structured array from :class:`~sage.data.primer.TimelineQuery`.
   * - ``save_parent_dir``
     - Root directory where the output files are written.
   * - ``noise_low_freq_cutoff``
     - Low-frequency cutoff in Hz stored as metadata; used downstream for whitening.
   * - ``corrupt_trim_length``
     - Seconds to trim from each end after resampling to remove edge artefacts.
   * - ``num_workers``
     - Number of parallel download threads. 64 is a reasonable default for a cluster
       node with fast internet.
   * - ``make_monolithic_file``
     - Concatenate all segments into one ``.bin`` file per detector instead of one
       file per segment.
   * - ``save_bin``
     - Write raw float32 binary (``True``) in addition to HDF5. The binary format is
       what :class:`~sage.data.noise.MemmapNoiseSampler` uses at training time.

Output files
------------

For each detector the downloader produces two files:

.. code-block:: text

    data_H1_O3b.h5    ← HDF5 archive with per-segment metadata
    data_H1_O3b.bin   ← flat float32 binary used by MemmapNoiseSampler

HDF5 structure
~~~~~~~~~~~~~~

.. code-block:: python

    import h5py
    f = h5py.File("data_L1_O3a.h5", "r")

    list(f.keys())               # ['segments']
    list(f["segments"].keys())   # ['00000', '00001', ...]

    # Per-segment attributes
    f["segments/00000"].attrs.keys()
    # ['detector', 'gps_end', 'gps_start', 'noise_low_freq_cutoff',
    #  'nsamples', 'old_sample_rate', 'run', 'sample_rate', 'trim']

Binary format
~~~~~~~~~~~~~

The ``.bin`` file is a flat ``float32`` array (little-endian) containing all segments
concatenated in order. Read it with NumPy:

.. code-block:: python

    import numpy as np

    arr = np.fromfile("data_L1_O3a.bin", dtype=np.float32)

.. note::

   The ``.bin`` file stores strain values scaled by PyCBC's ``DYN_RANGE_FAC`` (a large
   float to keep values in a comfortable numerical range). Divide by
   ``pycbc.DYN_RANGE_FAC`` to recover physical strain. The HDF5 and binary files are
   bit-identical after accounting for this factor.
