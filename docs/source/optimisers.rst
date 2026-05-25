Optimisers & Schedulers
========================

Sage uses standard PyTorch optimisers and schedulers, wrapped by
:class:`~sage.factory.schedulers.ManageScheduler` so the training loop does not need
to know whether a scheduler steps per-batch or per-epoch.

Setting up the optimiser
-------------------------

The O3b production run uses fused Adam with weight decay:

.. code-block:: python

    import torch.optim as optim

    optimiser = optim.Adam(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-6,
        fused=True,        # CUDA-fused kernel — faster than the default Python loop
    )

The ``fused=True`` flag enables a CUDA-fused Adam implementation that is 2–5× faster
than the standard Python loop, with identical numerics.  Requires a CUDA device.

Learning rate scheduler
------------------------

:class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` periodically resets
the learning rate to ``lr`` and decays it to ``eta_min`` via a cosine curve, then
repeats with a longer cycle:

.. code-block:: python

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    scheduler = CosineAnnealingWarmRestarts(
        optimiser,
        T_0=5,         # length of the first cycle (epochs)
        T_mult=2,      # cycle length multiplier; 2nd cycle = 10 epochs, 3rd = 20, …
        eta_min=1e-6,  # minimum learning rate at the bottom of each cosine
    )

The warm-restart strategy helps escape shallow local minima and explore different
regions of the loss landscape over the course of training.

ManageScheduler — step-mode adapter
-------------------------------------

:class:`~sage.factory.schedulers.ManageScheduler` wraps the scheduler and routes
:meth:`~sage.factory.schedulers.ManageScheduler.batch_step` and
:meth:`~sage.factory.schedulers.ManageScheduler.epoch_step` calls according to the
configured mode:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``mode``
     - When ``scheduler.step()`` is called
   * - ``"batch"`` (default)
     - Once per batch, unconditionally.  Appropriate for
       :class:`~torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` which
       interprets each ``step()`` call as one training step.
   * - ``"fractional"``
     - ``scheduler.step(epoch + batch/total_batches)`` — passes a fractional epoch
       value, useful for schedulers that accept continuous epoch inputs.
   * - ``"epoch"``
     - Once per epoch, via :meth:`~sage.factory.schedulers.ManageScheduler.epoch_step`.
   * - ``"metric"``
     - ``scheduler.step(val_loss)`` per epoch — for
       :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`.

.. code-block:: python

    from sage.factory.schedulers import ManageScheduler

    managed = ManageScheduler(scheduler, mode="batch")
    # Inside the training loop, batch_step is called automatically by SageUncompiledTraining

The training classes call ``managed.batch_step(nepoch, nbatch, num_iterations)``
at the end of every batch, so you do not need to call ``scheduler.step()`` manually.

AMP gradient scaler
--------------------

Mixed-precision training (``autocast=True``) requires a gradient scaler to avoid
underflow in float16 gradients:

.. code-block:: python

    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.autocast)

Pass ``scaler`` to :class:`~sage.factory.training.SageUncompiledTraining`. If
``cfg.autocast`` is ``False`` (e.g. for debugging on CPU), pass ``scaler=None`` — the
training loop handles the ``None`` case transparently.

Gradient clipping
-----------------

:class:`~sage.factory.training.SageUncompiledTraining` applies gradient norm clipping
after the backward pass using ``cfg.clip_norm`` (default ``1.0``):

.. code-block:: python

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_norm)

Clipping is applied *after* ``scaler.unscale_()`` when AMP is active, ensuring the
unscaled gradients are compared against the norm threshold.
