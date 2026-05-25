Complete Networks
==================

Both complete networks share the same frontend–backend architecture and differ only
in how the regression head is structured.

MSCNN1D_2DResNetCBAM
---------------------

:class:`~sage.architecture.network.MSCNN1D_2DResNetCBAM` outputs **point estimates**
— a single scalar per regression target. Used with
:class:`~sage.architecture.custom_losses.BCEWithPEregLoss`.

.. code-block:: python

    from sage.architecture.network import MSCNN1D_2DResNetCBAM

    model = MSCNN1D_2DResNetCBAM(
        frontend_filters=32,         # base filter count in ConvBlock
        frontend_kernel=64,          # base kernel size in ConvBlock
        backend_resnet_size=50,      # 18 | 34 | 50 | 101 | 152
        norm_type="instancenorm",    # "batchnorm" | "layernorm" | "instancenorm"
    ).to(dtype=cfg.dtype, device=cfg.device)

**Forward pass:**

.. code-block:: python

    ranking_stat, point_estimates = model(x)
    # x:               (B, n_det, compressed_samples)
    # ranking_stat:     (B, 1)           — raw logit for BCE classification
    # point_estimates:  (B, num_pe)      — one scalar per regression target

MSCNN1D_2DResNetCBAM_Heteroscedastic
--------------------------------------

:class:`~sage.architecture.network.MSCNN1D_2DResNetCBAM_Heteroscedastic` outputs
both the **predicted mean** and the **predicted log-variance** for each regression
target. The log-variance is learned alongside the mean, allowing the network to
express varying confidence about each prediction. Used with
:class:`~sage.architecture.custom_losses.BCEWithPEsigmaLoss`.

This is the model used in production (O3b run).

.. code-block:: python

    from sage.architecture.network import MSCNN1D_2DResNetCBAM_Heteroscedastic

    model = MSCNN1D_2DResNetCBAM_Heteroscedastic(
        frontend_filters=32,
        frontend_kernel=64,
        backend_resnet_size=50,
        norm_type="instancenorm",
    ).to(dtype=cfg.dtype, device=cfg.device, memory_format=torch.channels_last)

**Forward pass:**

.. code-block:: python

    ranking_stat, point_estimates = model(x)
    # ranking_stat:    (B, 1)
    # point_estimates: (B, 2 * num_pe)
    #   columns 0..num_pe-1     = predicted means  μ
    #   columns num_pe..2*num_pe-1 = predicted log-variances  log σ²

The predicted uncertainty ``σ = exp(0.5 * log_var)`` can be converted to a 1-sigma
confidence interval in physical units after unstandardisation.

Compiling with torch.compile
------------------------------

Both networks are compatible with ``torch.compile``. In production use
``mode="max-autotune"`` for maximum throughput:

.. code-block:: python

    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)

Compile adds a one-time warmup cost (typically 2–10 minutes) but reduces per-batch
inference time by 20–40% on A100/H100 hardware.

Parameter count
----------------

For the default configuration (``frontend_filters=32``, ``resnet50``, 2 detectors,
2 regression targets — ``tc`` and ``mchirp``):

.. code-block:: python

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Total parameters:     36,692,411
    # Trainable parameters: 36,692,411

Normalisation choice
---------------------

The ``norm_type`` argument selects the input normalisation layer applied before
the frontend:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``norm_type``
     - Behaviour
   * - ``"instancenorm"``
     - Normalises each sample independently across the time axis. Recommended —
       robust to batch-size variation and compatible with ``torch.compile``.
   * - ``"batchnorm"``
     - Normalises across the batch. Sensitive to batch size; less stable with
       ``autocast``.
   * - ``"layernorm"``
     - Normalises across all features of a sample. Rarely used for 1D signals.
