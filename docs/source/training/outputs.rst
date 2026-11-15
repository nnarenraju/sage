What Training Writes
====================

Training writes everything into the run directory, next to ``run.yaml`` and
``prior.yaml``. After a short run, it holds:

.. code-block:: text

   my_run/
     run.yaml
     prior.yaml
     checkpoints/
       best.pt
       ema.pt
       ema_calibrated.pt
       epoch_0000.pt
       epoch_0001.pt
       latest.pt
       progress.json
     results/
       losses.h5
       validation.h5
       final_check.json
       final_check_estimate.json
     logs/
       run.log
     resolved/
       run.yaml
       prior.yaml
       derived.yaml
       versions.yaml
       fiducial/
     flags/

Checkpoints
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - ``checkpoints/``
     - What it holds
   * - ``latest.pt``
     - The last saved epoch, which a resumed training starts from.
   * - ``epoch_NNNN.pt``
     - The last epochs, as many as ``train.keep_checkpoints``.
   * - ``best.pt``
     - The epoch with the lowest validation loss.
   * - ``ema.pt``
     - The average of the weights over training.
   * - ``ema_calibrated.pt``
     - The averaged weights with their batch-norm statistics measured again,
       written by the final check.
   * - ``progress.json``
     - How far training has got.

Without ``ema``, there is no ``ema.pt`` or ``ema_calibrated.pt``.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - ``results/``
     - What it holds
   * - ``losses.h5``
     - The training and validation losses of every epoch, and their parts.
   * - ``validation.h5``
     - The network's outputs on the validation windows, for every validation
       pass and the final check. ``sage plot`` draws its figures from it.
   * - ``final_check.json``
     - The final check's losses, and the checkpoint with the lower one.
   * - ``final_check_estimate.json``
     - How long the final check is expected to take.

``final_check.json`` reads:

.. code-block:: json

   {
    "samples": 16,
    "best.pt": {
     "epoch": 1,
     "loss": {
      "total": 0.6945825815200806,
      "classification": 0.6931493878364563,
      "regression_delta_tc": 0.013082402758300304,
      "regression_mchirp": -0.07201029360294342,
      "coupling": 0.34555888175964355
     }
    },
    "ema_calibrated.pt": {
     "epoch": 1,
     "loss": {
      "total": 0.7073580026626587,
      "classification": 0.7059579491615295,
      "regression_delta_tc": 0.015463508665561676,
      "regression_mchirp": -0.07282368093729019,
      "coupling": 0.3373645544052124
     },
     "batchnorm_layers_recalibrated": 94,
     "calibration_batches": 2
    },
    "lower_total_loss": "best.pt"
   }

The record of the run
---------------------

``resolved/`` is the run's fixed record, written before the first
checkpoint:

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - ``resolved/``
     - What it holds
   * - ``run.yaml``, ``prior.yaml``
     - The settings and prior the run trains with.
   * - ``derived.yaml``
     - Values Sage computes from the settings, such as the multirate bins
       and the scaling of the estimated parameters.
   * - ``versions.yaml``
     - The Sage commit and library versions of every job that trained the
       run.
   * - ``fiducial/``
     - A copy of the fiducial ASDs the run whitens with.

A clone also has ``resolved/clone.yaml``, saying which run it came from.

Other folders
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - Folder
     - What it holds
   * - ``logs/run.log``
     - Everything training prints. ``sage log`` shows it.
   * - ``logs/jobs/``
     - The output of each cluster job, from ``sage slurm-chain``.
   * - ``flags/``
     - ``running`` while training runs, ``stop`` after ``sage stop``, and
       ``checked`` after ``sage check``.
   * - ``mining/bank.h5``
     - With mining on, the noise windows mining has found.
