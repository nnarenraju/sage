Troubleshooting
================

.. dropdown:: CUDA out of memory (OOM)

   **Symptoms:** ``torch.cuda.OutOfMemoryError`` during training or data generation.

   **Fixes (try in order):**

   1. Reduce ``batch_size`` in ``RunCFG`` (halving it roughly halves peak VRAM usage).
   2. Disable ``torch.backends.cudnn.benchmark = True`` — it allocates extra workspace.
   3. Set ``autocast = True`` in ``RunCFG`` to use float16 during the forward pass,
      which cuts activation memory roughly in half.
   4. Reduce ``prefetch`` in ``MemmapNoiseSampler`` (each prefetch slot holds a full
      batch on the GPU).
   5. If using ResNet-101 or ResNet-152, switch to ResNet-50 (``backend_resnet_size=50``).

.. dropdown:: torch.compile / Dynamo errors

   **Symptoms:** Errors containing ``torch._dynamo``, ``graph break``, or
   ``TorchDynamoException`` at the start of training.

   **Fixes:**

   - Set ``torch._dynamo.config.verbose = True`` temporarily to identify the break point.
   - Ensure the model is moved to device *before* calling ``torch.compile``.
   - Use ``dynamic=True`` in the compile call if batch size varies between training
     and validation.
   - As a last resort, remove ``torch.compile`` for debugging — the uncompiled model
     is functionally identical.

.. dropdown:: lalsuite or pycbc import errors

   **Symptoms:** ``ModuleNotFoundError: No module named 'lal'`` or similar at import.

   **Fix:** LALSuite and PyCBC are optional dependencies not installed by default.
   Install them via conda (strongly recommended over pip):

   .. code-block:: bash

      conda install -c conda-forge lalsuite pycbc

   Or via pip (may fail on some systems):

   .. code-block:: bash

      pip install sage-gw[lal]

   These are only required for waveform validation against LALSuite and for PyCBC
   benchmarking. Core training does not need them.

.. dropdown:: Data download timeouts or failures

   **Symptoms:** ``TimeoutError``, ``ConnectionError``, or partial downloads from GWOSC.

   **Fixes:**

   - Increase ``max_download_retries`` and ``retry_delay`` in ``DataReleaseDownloader``.
   - Run from a cluster node with a stable network connection — GWOSC data is large
     (O3b is ~several TB).
   - Check GWOSC status at `gwosc.org <https://gwosc.org>`__ for service outages.
   - If a segment fails repeatedly, use ``auto_clean_empty_timelines=True`` in
     ``TimelineQuery`` to skip it.

.. dropdown:: NaN losses during training

   **Symptoms:** Loss becomes ``nan`` after a few batches.

   **Fixes (try in order):**

   1. Lower the learning rate (e.g. ``lr=1e-4`` instead of ``2e-4``).
   2. Verify gradient clipping is active — ``cfg.clip_norm`` should be ``1.0``.
   3. Check that ``autocast`` and ``GradScaler`` are both enabled or both disabled —
      mixing states can cause scale underflow.
   4. Temporarily set ``torch.autograd.set_detect_anomaly(True)`` to find which
      operation produces the NaN.
   5. Inspect the noise data: a corrupt ``.bin`` file can produce all-zero or
      all-inf frames. Re-run the download step for affected segments.

.. dropdown:: Low GPU utilisation during training

   **Symptoms:** ``nvidia-smi`` shows GPU at < 50% utilisation.

   **Fixes:**

   - Increase ``prefetch`` in ``MemmapNoiseSampler`` (default 8 is usually sufficient,
     but disk-bound machines may need more).
   - Ensure ``num_workers`` is set in ``DataReleaseDownloader`` for parallel downloads.
   - Compile the model with ``torch.compile(mode="max-autotune")`` — the first epoch
     will be slow during kernel tuning, but steady-state throughput improves
     significantly afterwards.
   - Move the ``.bin`` data files to a fast local SSD rather than a network filesystem.

.. dropdown:: Checkpoint fails to load

   **Symptoms:** ``KeyError`` or ``RuntimeError`` when restoring from a checkpoint.

   **Common causes and fixes:**

   - **Model architecture mismatch** — the checkpoint was saved with a different
     ``backend_resnet_size`` or ``norm_type``. Rebuild the model with the same
     parameters before loading.
   - **Compiled vs. uncompiled state dict** — ``torch.compile`` wraps the model in a
     ``_orig_mod`` prefix. Load with ``strict=False`` or unwrap with
     ``model._orig_mod`` if mixing compiled and uncompiled saves.
   - **Scaler state missing** — checkpoints saved before the scaler fix (see changelog
     v0.1.0) will not have a ``scaler`` key. Pass ``strict=False`` or set
     ``scaler=None`` and reinitialise.
