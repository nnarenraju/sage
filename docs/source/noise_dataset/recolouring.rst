Recolouring Noise & Returning FD Batches
==========================================

Recolouring is a training-time augmentation that randomly replaces the spectral shape
of a noise batch with one drawn from the precomputed recolouring PSD library. This
exposes the network to a wide variety of noise PSDs without requiring separate
recordings for each configuration.

The full pipeline — sample a batch, optionally recolour, return in the frequency domain
— is handled by :class:`~sage.data.noise.recolour.RecolourPostprocess` combined with
:class:`~sage.data.noise.MemmapNoiseSampler`.

Setting up the recolouring postprocess
----------------------------------------

.. code-block:: python

    from sage.data.noise import MemmapNoiseSampler
    from sage.data.noise import recolour

    files = [
        "./data_release/data_H1_O3a.bin",
        "./data_release/data_L1_O3a.bin",
    ]

    seq_len = int(
        (data_cfg.sample_length + 2 * data_cfg.corrupted_length) * data_cfg.sample_rate
    )

    postprocess = recolour.RecolourPostprocess(
        data_dir=data_cfg.data_dir,
        detectors=cfg.detectors,
        seq_len=seq_len,
        sample_rate=data_cfg.sample_rate,
        p_recolour=0.37,           # 37% of batches are recoloured
        device=cfg.device,
    )

    noise_sampler = MemmapNoiseSampler(
        files,
        seq_len=seq_len,
        device=cfg.device,
        batch_size=cfg.batch_size,
        prefetch=3,
        postprocess_fn=postprocess,
        cfg=cfg(),
        data_cfg=data_cfg(),
    )

Sampling a batch
-----------------

.. code-block:: python

    noise_batch = noise_sampler()
    print(noise_batch.shape)   # torch.Size([64, 2, 16385])

The output is a complex frequency-domain tensor of shape
``(batch_size, n_detectors, n_freq_bins)`` on ``cfg.device``.
The ``n_freq_bins`` dimension spans ``[0, sample_rate/2]`` with spacing ``delta_f``.

With ``p_recolour=0.37``, each call to ``noise_sampler()`` independently decides
whether to apply recolouring. When recolouring fires, the batch is:

1. Whitened using its own per-segment PSD.
2. Re-coloured with a PSD drawn uniformly from the recolouring library.
3. Returned in the frequency domain.

When recolouring does not fire, the batch is whitened with the fiducial PSD and
returned directly.

Performance
-----------

With recolouring PSDs loaded into CPU RAM (not VRAM), average batch latency remains
around **11 ms** for a batch of 64 with 2 detectors:

.. code-block:: python

    import time

    # Warm up
    for _ in range(100):
        noise_sampler.sample_batch()

    t0 = time.perf_counter()
    for _ in range(1000):
        noise_sampler.sample_batch()
    t1 = time.perf_counter()

    print("Avg batch latency:", (t1 - t0) / 1000)
    # Avg batch latency: 0.011 s

    noise_sampler.shutdown()

.. note::

   Storing all recolouring PSDs on VRAM shaves roughly 2–3 ms per batch but is
   typically impractical because of VRAM capacity. The CPU path is fast enough for
   production training.
