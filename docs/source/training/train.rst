Training a Run
==============

``sage train`` trains the run's network, from ``run.yaml`` and
``prior.yaml``:

.. code-block:: console

   $ sage train

It trains on the device ``train.device`` names, a GPU for every preset.

Each epoch
----------

Every epoch trains on ``train.samples_per_epoch`` new samples: signals drawn
from the prior and added to noise windows read from the strain, so every
sample is new. After its training pass, an epoch:

* validates the network, every ``validation.every_n_epochs`` epochs and
  after the last one;
* with mining on, mines the noise for windows the network mistakes for a
  signal, at the epochs ``mining`` sets, to train on in later epochs;
* saves a checkpoint, and updates ``best.pt`` when the validation loss is
  the lowest so far.

A short run of two epochs prints:

.. code-block:: console

   $ sage train
   23:08:37 | INFO    | Run my_run from epoch 0, seed 150914
   23:08:42 | INFO    | Epoch 0 | training: 1 iterations, batch 4
   23:08:44 | INFO    |   epoch 0 | 100% (1/1) | loss 0.6914 | 0.6 it/s | eta 0s
   23:08:44 | INFO    | Epoch 0 | train loss 0.69140 | components [0.69005, 0.01943, -0.08860, 0.34002] | lr 2.000e-04 | 0.6 it/s | 1s
   23:08:44 | INFO    | Epoch 0 | validating: 1 iterations
   23:08:44 | INFO    | Epoch 0 | val   loss 0.69424 | components [0.69315, -0.05351, -0.07519, 0.34640] | 0s
   23:08:45 | INFO    | Epoch 1 | training: 1 iterations, batch 4
   23:08:47 | INFO    |   epoch 1 | 100% (1/1) | loss 0.6881 | 0.4 it/s | eta 0s
   23:08:47 | INFO    | Epoch 1 | train loss 0.68813 | components [0.68730, -0.07675, -0.08876, 0.33156] | lr 2.000e-04 | 0.4 it/s | 2s
   23:08:47 | INFO    | Epoch 1 | validating: 1 iterations
   23:08:48 | INFO    | Epoch 1 | val   loss 0.69423 | components [0.69315, -0.05464, -0.07504, 0.34556] | 0s
   23:08:50 | INFO    | Epoch 0 | validating: 4 iterations
   23:08:52 | INFO    | Epoch 0 | val   loss 0.69458 | components [0.69315, 0.01308, -0.07201, 0.34556] | 1s
   23:08:52 | INFO    | Epoch 0 | validating: 4 iterations
   23:08:53 | INFO    | Epoch 0 | val   loss 0.70736 | components [0.70596, 0.01546, -0.07282, 0.33736] | 1s
   23:08:53 | INFO    | Final check on 16 validation samples: best.pt (epoch 1) loss 0.69458; ema_calibrated.pt (epoch 1) loss 0.70736
   Trained to the end. Next: sage status my_run

Each loss line gives the total, then its parts: the classification, one
regression for each estimated parameter (``delta_tc``, then ``mchirp``), and
the coupling between them. ``lr`` is the learning rate.

The final check
---------------

After the last epoch, the final check compares two versions of the network
on a larger validation set, ``validation.final_samples`` windows:

* ``best.pt``, the epoch with the lowest validation loss;
* ``ema_calibrated.pt``, the average of the weights over training, with its
  batch-norm statistics measured again.

It names the one with the lower loss, in its last log line and in
``results/final_check.json``. Its two validation passes are the last two in
the output above. A run without ``ema`` checks ``best.pt`` alone.

Stopping and resuming
---------------------

Training resumes where it stopped: run ``sage train`` again, and it carries
on from the last saved epoch, with the same random draws as if it had never
stopped. A run that has finished returns at once:

.. code-block:: console

   $ sage train
   Trained to the end. Next: sage status my_run

To end training at a time limit, ``--stop-after`` saves and exits before an
epoch that would not end in time. It also leaves the final check for the
next job if the check would not end in time:

.. code-block:: console

   $ sage train --stop-after 47h

:doc:`cluster` shows how this chains jobs on a cluster, and :doc:`follow`
shows ``sage stop``, which stops training after the current epoch.

The settings are fixed
----------------------

The first checkpoint fixes the run's settings. ``sage train`` writes them to
``resolved/`` before it starts, and checks them each time it resumes. To
change a run that has started, clone it: :doc:`../runs/cloning`.

``sage train --help`` shows every option:

.. code-block:: console

   $ sage train --help
   usage: sage train [-h] [--stop-after STOP_AFTER] [directory]

   Train the run, resuming from its latest checkpoint.

   positional arguments:
     directory             run directory, or a directory inside it (default: here)

   options:
     -h, --help            show this help message and exit
     --stop-after STOP_AFTER
                           save and exit before an epoch that would not end in time, e.g. 47h

What training writes is on :doc:`outputs`.
