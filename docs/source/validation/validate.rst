Validating a Checkpoint
=======================

``sage validate`` scores one stored checkpoint on the run's final validation
set: ``validation.final_samples`` windows, the same ones the final check
uses. It validates ``best.pt`` unless told otherwise:

.. code-block:: console

   $ sage validate
   01:43:19 | INFO    | Epoch 0 | validating: 4 iterations
   01:44:20 | INFO    | Epoch 0 | val   loss 0.69458 | components [0.69315, 0.01308, -0.07201, 0.34556] | 1m 00s
   Validated best.pt (epoch 1) on 16 validation samples: total 0.69458, classification 0.69315, regression_delta_tc 0.01308, regression_mchirp -0.07201, coupling 0.34556
   Added to results/validate.h5 as group validate_best
   Next: sage plot

``--from-epoch`` validates another stored epoch, and ``--ema-calibrated``
validates ``ema_calibrated.pt``:

.. code-block:: console

   $ sage validate --from-epoch 0
   Validated epoch_0000.pt (epoch 0) on 16 validation samples: total 0.69458, classification 0.69315, regression_delta_tc 0.01293, regression_mchirp -0.07196, coupling 0.34640
   Added to results/validate.h5 as group validate_epoch_0000
   Next: sage plot

   $ sage validate --ema-calibrated
   Validated ema_calibrated.pt (epoch 1) on 16 validation samples: total 0.70736, classification 0.70596, regression_delta_tc 0.01546, regression_mchirp -0.07282, coupling 0.33736
   Added to results/validate.h5 as group validate_ema_calibrated
   Next: sage plot

Only epochs still stored can be validated: the last ``train.keep_checkpoints``
of them, and ``best.pt``.

Each checkpoint gets its own group in ``results/validate.h5``, so its
figures can be drawn like any validation pass:

.. code-block:: console

   $ sage plot --group validate_epoch_0000

``sage validate --help`` shows every option:

.. code-block:: console

   $ sage validate --help
   usage: sage validate [-h] [--from-epoch FROM_EPOCH] [--ema-calibrated] [directory]

   Validate one stored checkpoint on the run's final validation set.

   positional arguments:
     directory             run directory, or a directory inside it (default: here)

   options:
     -h, --help            show this help message and exit
     --from-epoch FROM_EPOCH
                           a stored epoch instead of best.pt
     --ema-calibrated      ema_calibrated.pt instead of best.pt

The two options each name a checkpoint, so only one can be given:

.. code-block:: console

   $ sage validate --from-epoch 1 --ema-calibrated
   error: --from-epoch and --ema-calibrated each name a checkpoint; give one
   fix: sage validate --ema-calibrated
