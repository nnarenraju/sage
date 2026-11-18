Validation During Training
==========================

While a run trains, Sage checks the network on validation windows: signals
and noise made the same way as for training, but with the noise from
``noise.validation``, so the network is checked on noise it never trained
on. For every preset, that is O3a, while training uses O3b.

Every pass uses the same validation windows, so the losses of different
epochs can be compared, and the epoch with the lowest one is kept as
``best.pt``.

The validation settings
-----------------------

The ``validation`` section of ``run.yaml`` sets the passes. For ``quick``:

.. code-block:: yaml

   validation:
     samples: 6400
     every_n_epochs: 2
     final_samples: 25600
     seed: 170817
     snr:
       name: halfnorm
       scale: 4.0
       loc: 5.0

.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: wrap

   * - Setting
     - What it sets
   * - ``samples``
     - How many windows each validation pass scores.
   * - ``every_n_epochs``
     - How often a pass runs. The last epoch is always validated.
   * - ``final_samples``
     - How many windows the final check scores.
   * - ``seed``
     - The seed the validation windows are drawn with.
   * - ``snr``
     - How loud the validation signals are.

What a pass stores
------------------

Each pass adds its losses to ``results/losses.h5`` and the network's outputs
to ``results/validation.h5``, in a group of its own:

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - Group
     - Written by
   * - ``epoch_NNNN``
     - The validation pass after epoch ``NNNN``.
   * - ``final_best``, ``final_ema``
     - The final check, for ``best.pt`` and ``ema_calibrated.pt``.

``sage validate`` keeps its passes apart, in ``results/validate.h5``:
:doc:`validate` explains them. :doc:`figures` draws figures from any of
these groups.
