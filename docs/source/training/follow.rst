Following Training
==================

Three commands follow a run while it trains, or after: ``sage status``,
``sage log`` and ``sage stop``.

How far a run has got
---------------------

``sage status`` shows how many epochs are trained, the latest losses, and
the final check once it has run:

.. code-block:: console

   $ sage status
   my_run: 2 of 2 epochs trained
   Training loss 0.68813 (epoch 1)
   Validation loss 0.69423 (epoch 1); lowest 0.69423 (epoch 1, best.pt)
   Final check, best.pt (epoch 1): loss 0.69458 on 16 samples
   Final check, ema_calibrated.pt (epoch 1): loss 0.70736 on 16 samples
   Log last written 2026-11-14 22:08:53 UTC, 1:36:30 ago
   Next: sage plot

While the run trains, it also estimates when training will finish, final
check included, and how long the final check will take, so a cluster job can
be left for it.

Times are in UTC unless ``--timezone`` names another zone:

.. code-block:: console

   $ sage status --help
   usage: sage status [-h]
                      [--timezone {utc,uk,europe-central,europe-eastern,us-eastern,us-central,us-mountain,us-pacific,india,china,japan,australia-eastern,australia-western}]
                      [directory]

   Show how far the run has trained and its latest losses.

   positional arguments:
     directory             run directory, or a directory inside it (default: here)

   options:
     -h, --help            show this help message and exit
     --timezone {utc,uk,europe-central,europe-eastern,us-eastern,us-central,us-mountain,us-pacific,india,china,japan,australia-eastern,australia-western}
                           time zone of the times shown (default: utc)

The log
-------

Everything training prints also goes to ``logs/run.log``. ``sage log`` shows
its last lines:

.. code-block:: console

   $ sage log --lines 4
   2026-11-14 23:08:52 | INFO    | sage.factory.validation:243 | Epoch 0 | val   loss 0.69458 | components [0.69315, 0.01308, -0.07201, 0.34556] | 1s
   2026-11-14 23:08:52 | INFO    | sage.factory.validation:178 | Epoch 0 | validating: 4 iterations
   2026-11-14 23:08:53 | INFO    | sage.factory.validation:243 | Epoch 0 | val   loss 0.70736 | components [0.70596, 0.01546, -0.07282, 0.33736] | 1s
   2026-11-14 23:08:53 | INFO    | sage.factory.train_run:962 | Final check on 16 validation samples: best.pt (epoch 1) loss 0.69458; ema_calibrated.pt (epoch 1) loss 0.70736
   Next: sage status

``--follow`` keeps printing new lines as training writes them, until Ctrl-C:

.. code-block:: console

   $ sage log --follow

.. code-block:: console

   $ sage log --help
   usage: sage log [-h] [--lines LINES] [--follow] [directory]

   Print the end of the run's log, or follow it.

   positional arguments:
     directory      run directory, or a directory inside it (default: here)

   options:
     -h, --help     show this help message and exit
     --lines LINES  lines to print (default: 20)
     --follow       keep printing new lines until Ctrl-C

Stopping training
-----------------

``sage stop`` stops training once its current epoch is saved. Later jobs in a
cluster job chain then end at once, without training:

.. code-block:: console

   $ sage stop
   Wrote flags/stop: training stops once its current epoch is saved, and later jobs do not start. To train on: rm flags/stop && sage train
   Next: sage status

``sage status`` shows the stop, and ``sage train`` refuses to start until the
flag is removed:

.. code-block:: console

   $ sage status
   my_run: training has not started
   Flags: stop (since 2026-11-14 23:47:04 UTC)
   Stopped by sage stop; to train on: rm flags/stop && sage train
   Next: sage train

   $ sage train
   error: my_run was stopped by sage stop
   fix: rm flags/stop && sage train

To train on, remove the flag and train again; training resumes from the last
saved epoch.

.. code-block:: console

   $ sage stop --help
   usage: sage stop [-h] [directory]

   Stop the run's training after its current epoch.

   positional arguments:
     directory   run directory, or a directory inside it (default: here)

   options:
     -h, --help  show this help message and exit
