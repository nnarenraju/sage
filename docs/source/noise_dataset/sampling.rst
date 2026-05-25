Sampling Noise Batches
=======================

Sage provides two noise samplers: a single-segment sampler for exploration and PSD
estimation, and a high-throughput batched sampler backed by a background prefetch thread
for use inside training loops.

Single-segment sampler
-----------------------

:class:`~sage.data.noise.MemmapSingleNoiseSampler` reads one segment at a time from a
``.bin`` file. It automatically weights segments proportionally to their duration so that
sampling is approximately uniform in GPS time.

.. code-block:: python

    from sage.data.noise import MemmapSingleNoiseSampler

    sampler = MemmapSingleNoiseSampler("./data_release/data_L1_O3a.bin")

    # Returns a numpy float32 array of length 40960
    segment = sampler(40960)

To get a PyTorch tensor instead:

.. code-block:: python

    sampler = MemmapSingleNoiseSampler(
        "./data_release/data_L1_O3a.bin",
        return_tensor=True,
    )
    segment = sampler(40960)   # torch.Tensor, on CPU

Batched sampler
---------------

:class:`~sage.data.noise.MemmapNoiseSampler` spawns a background prefetch thread and
keeps a queue of pre-loaded batches ready for the GPU. Pass multiple files (one per
detector) — the sampler returns batches of shape ``(batch_size, n_detectors, seq_len)``.

.. code-block:: python

    from sage.data.noise import MemmapNoiseSampler

    files = [
        "./data_release/data_H1_O3a.bin",
        "./data_release/data_L1_O3a.bin",
    ]

    seq_len = int((data_cfg.sample_length + 2 * data_cfg.corrupted_length) * data_cfg.sample_rate)

    sampler = MemmapNoiseSampler(
        files,
        seq_len=seq_len,
        device="cuda:0",
        batch_size=64,
        prefetch=3,
    )

    noise_batch = sampler.sample_batch(64)
    print(noise_batch.shape)   # torch.Size([64, 2, 16385])

Always call ``sampler.shutdown()`` when done to cleanly stop the prefetch thread:

.. code-block:: python

    sampler.shutdown()

Performance
-----------

With prefetch enabled, average batch latency is around **10 ms** for a batch of 64
at 2048 Hz with 2 detectors, measured on a GPU cluster node:

.. code-block:: python

    import time

    # Warm up
    for _ in range(100):
        sampler.sample_batch(64)

    t0 = time.perf_counter()
    for _ in range(1000):
        sampler.sample_batch(64)
    t1 = time.perf_counter()

    print("Avg batch latency:", (t1 - t0) / 1000)
    # Avg batch latency: 0.00997 s
