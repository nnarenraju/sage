Training
=========

:class:`~sage.factory.training.SageUncompiledTraining` is the reference training
loop used in all production runs. It constructs each batch explicitly, injects
signals into random noise positions, preprocesses the combined batch, runs the
forward pass, and updates the model.

Initialisation
--------------

.. code-block:: python

    from sage.factory.training import SageUncompiledTraining

    train_sage = SageUncompiledTraining(
        signal_sampler,      # e.g. IMRPhenomPv2 with SNR rescaler
        noise_sampler,       # e.g. MemmapNoiseSampler with RecolourPostprocess
        processor,           # Preprocessor([FiducialWhitening(), MultirateSampler(...)])
        model,               # MSCNN1D_2DResNetCBAM_Heteroscedastic (compiled or not)
        loss_function,       # BCEWithPEsigmaLoss(...)
        optimiser,           # optim.Adam(...)
        scheduler,           # CosineAnnealingWarmRestarts(...)
        scaler,              # torch.amp.GradScaler(...) or None
        num_iterations=cfg.training_iterations,   # batches per epoch
        num_epochs=cfg.num_epochs,                # total epochs
        scheduler_mode="batch",                   # step scheduler every batch
    )

Constructor arguments
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``signal_sampler``
     - Any callable returning ``(signal_data, signal_targets)``. The number
       of signal samples drawn per call is ``int(batch_size * class_balance)``.
   * - ``noise_sampler``
     - Any callable returning ``(noise_data, noise_targets)``. Must produce
       ``batch_size`` samples.
   * - ``processor``
     - Applied to the combined batch after signal injection.
   * - ``model``
     - The network (may be a ``torch.compile``-wrapped model).
   * - ``loss_function``
     - Returns a stacked loss-component tensor; its ``num_components`` attribute
       sets the shape of the loss tracking buffer.
   * - ``optimiser``
     - Any :class:`torch.optim.Optimizer`.
   * - ``scheduler``
     - Any ``lr_scheduler``; wrapped internally by
       :class:`~sage.factory.schedulers.ManageScheduler`.
   * - ``scaler``
     - :class:`torch.amp.GradScaler` instance, or ``None`` to disable AMP.
   * - ``num_iterations``
     - Number of gradient steps per epoch. In the O3b run:
       ``int(2_000_000 / batch_size)`` — equivalent to 2 M samples per epoch.
   * - ``num_epochs``
     - Total number of epochs. Pre-allocates the ``loss_components`` tracking buffer.
   * - ``scheduler_mode``
     - ``"batch"`` (default) or ``"epoch"`` — controls when
       :class:`~sage.factory.schedulers.ManageScheduler` steps the scheduler.

What happens inside each batch
--------------------------------

The ``forward`` method (called via ``train_sage(nepoch=i)``) runs
``num_iterations`` batches per epoch. For each batch:

1. **Sample** — ``signal_sampler()`` and ``noise_sampler()`` produce
   ``S = batch_size * class_balance`` signals and ``B = batch_size`` noise windows.
2. **Inject** — ``S`` signals are injected into ``S`` randomly chosen positions of
   the ``B``-size noise batch. The injection positions are sampled without replacement
   via ``torch.randperm``.
3. **Preprocess** — ``processor(x)`` whitens and multirate-compresses the batch.
4. **Forward pass** — the model runs under ``torch.autocast`` if ``cfg.autocast`` is
   enabled, producing ``(ranking_stat, point_estimates)``.
5. **Loss** — ``loss_function(out, targets)`` returns a stacked component tensor.
6. **Backward** — ``loss[0].backward()`` with optional AMP scaling.
7. **Clip** — gradient norms clipped to ``cfg.clip_norm``.
8. **Step** — optimiser and scheduler updated.
9. **Accumulate** — ``loss_components[nepoch] += loss.detach()``.

Running the training loop
--------------------------

.. code-block:: python

    for nepoch in range(cfg.num_epochs):
        train_sage(nepoch=nepoch)

``train_sage(nepoch=i)`` runs one full epoch and updates ``train_sage.loss_components[i]``
with the per-component average loss.

Inspecting loss components
---------------------------

After each epoch, ``loss_components`` contains the mean losses for all completed epochs:

.. code-block:: python

    print(train_sage.loss_components)
    # tensor([[0.705, 0.689, 0.054],   ← epoch 0: [total, bce, reg]
    #         [0.477, 0.461, 0.054],   ← epoch 1
    #         ...], device='cuda:0')

For ``BCEWithPEsigmaLoss`` with 2 PE targets, the shape is ``(num_epochs, 4)``:
``[total, bce, nll_reg, coupling]``.

Logging losses to HDF5
------------------------

The production run uses :class:`~sage.core.logger.HDF5LossLogger` to persist losses
to disk:

.. code-block:: python

    from sage.core.logger import HDF5LossLogger

    logger = HDF5LossLogger(
        path=os.path.join(cfg.export_dir, "losses.h5"),
        num_epochs=cfg.num_epochs,
        num_components=train_sage.loss_function.num_components,
    )

    for nepoch in range(cfg.num_epochs):
        train_sage(nepoch=nepoch)
        logger.log(train_sage.loss_components, nepoch, split="training")

Hard-mining variant
--------------------

:class:`~sage.factory.training.SageHardMiningTraining` extends the standard loop by
periodically replacing a fraction of background slots with pre-mined high-ranking
noise windows (false-alarm candidates the model currently struggles to reject):

.. code-block:: python

    from sage.factory.training import SageHardMiningTraining

    train_sage = SageHardMiningTraining(
        ...,                          # same as SageUncompiledTraining
        miner=miner,                  # HardSampleMiner or subclass
        hard_noise_buffer=buffer,     # HardNoiseBuffer — filled by the miner
        hard_noise_frac=0.15,         # 15% of background slots → hard noise
        mine_every_n_epochs=5,        # re-mine every 5 epochs
    )

See :doc:`noise_dataset/sampling` for details on the mining strategies.
