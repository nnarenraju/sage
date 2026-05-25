Validation
===========

:class:`~sage.factory.validation.SageUncompiledValidation` mirrors the training loop
but runs the model in ``eval`` / ``inference_mode`` and saves full per-epoch diagnostic
outputs to an HDF5 file for offline analysis.

Unlike training, validation also:

* Calls ``signal_sampler(return_theta=True)`` to obtain the raw waveform parameter
  tensor ``θ`` alongside the processed targets — needed for parameter-recovery plots.
* Unstandardises predicted means back to physical units using the sampler's
  normalisation buffers.
* Converts the predicted log-variance to 1-sigma uncertainties (``σ = exp(0.5 * log_var)``).
* Writes all outputs per epoch to ``{export_dir}/validation_data.h5``.

Initialisation
--------------

.. code-block:: python

    from sage.factory.validation import SageUncompiledValidation

    validate_sage = SageUncompiledValidation(
        validation_signal_sampler,   # separate sampler with different seed
        validation_noise_sampler,    # no recolouring for validation
        processor,                   # same Preprocessor as training
        model,                       # shared model reference
        loss_function,               # same loss function
        num_iterations=cfg.validation_iterations,  # int(200_000 / batch_size)
        num_epochs=cfg.num_epochs,
    )

The validation signal and noise samplers should use **different seeds** from the
training samplers to ensure the validation set contains different waveform parameters
and noise windows:

.. code-block:: python

    # Training: seed=150914
    # Validation: seed=170817  (GW170817 date — easy to remember)

Running validation
-------------------

.. code-block:: python

    validate_sage(nepoch=nepoch)

The O3b run validates every 5 epochs plus epoch 0:

.. code-block:: python

    for nepoch in range(cfg.num_epochs):
        train_sage(nepoch=nepoch)
        logger.log(train_sage.loss_components, nepoch, split="training")

        if (nepoch + 1) % 5 == 0 or nepoch == 0:
            validate_sage(nepoch=nepoch)
            logger.log(validate_sage.loss_components, nepoch, split="validation")

What is saved to HDF5
----------------------

For each validated epoch, ``{export_dir}/validation_data.h5`` gains a group
``epoch_{nepoch:04d}`` containing four gzip-compressed datasets:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Dataset
     - Shape
     - Contents
   * - ``network_output``
     - ``(N, 1 + 2·num_pe)``
     - Ranking statistic (column 0), predicted parameter means in physical units
       (columns 1…num_pe), predicted 1-σ uncertainties in physical units
       (columns num_pe+1…).
   * - ``network_target``
     - ``(N, num_pe + 1)``
     - Regression targets and class labels (0 = noise, 1 = signal).
   * - ``signal_params``
     - ``(N, num_all_params)``
     - Full raw waveform parameter tensor ``θ`` for every batch item. Noise
       samples have zeros here — use ``network_target[:, -1]`` to identify
       signal rows.
   * - ``signal_idx``
     - ``(num_iter, S)``
     - Batch positions where signals were injected, one row per iteration.
       Used to align ``signal_params`` rows with the combined batch.

Reading the HDF5 output
------------------------

.. code-block:: python

    import h5py
    import numpy as np

    with h5py.File("run_export/validation_data.h5", "r") as f:
        epoch = "epoch_0004"
        ranking  = f[f"{epoch}/network_output"][:, 0]
        mu_tc    = f[f"{epoch}/network_output"][:, 1]
        sigma_tc = f[f"{epoch}/network_output"][:, 3]   # after num_pe columns
        labels   = f[f"{epoch}/network_target"][:, -1]
        theta    = f[f"{epoch}/signal_params"][:]

    # Separate signal and noise rows
    signal_mask = labels == 1
    noise_mask  = labels == 0

    signal_rankings = ranking[signal_mask]
    noise_rankings  = ranking[noise_mask]

Downstream diagnostics
-----------------------

The HDF5 output drives all post-training diagnostic plots in
:mod:`sage.plotting`:

* **ROC curve** — ``ranking`` vs. ``labels`` across many injections.
* **Efficiency curve** — detection fraction as a function of injection SNR.
* **Parameter recovery** — predicted ``mu_tc`` / ``sigma_tc`` vs. true ``tc`` from
  ``signal_params``.
* **Loss curves** — read from the HDF5 loss logger alongside the validation file.

Checkpointing
--------------

The production run checkpoints the best model after each validation pass based on
the total validation loss:

.. code-block:: python

    from sage.utils.checkpoint import CheckpointManager

    ckpt_mgr = CheckpointManager(
        cfg=cfg,
        data_cfg=data_cfg,
        model=model,
        optimizer=optimiser,
        scheduler=scheduler,
        scaler=scaler,
    )

    val_loss = validate_sage.loss_components[nepoch][0].item()
    ckpt_mgr.save(epoch=nepoch, val_loss=val_loss)

:class:`~sage.utils.checkpoint.CheckpointManager` saves the full model state dict,
optimiser state, scheduler state, and scaler state to ``{export_dir}/checkpoints/``,
keeping the best checkpoint (by val_loss) and the most recent one.
