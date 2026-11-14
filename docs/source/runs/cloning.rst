Cloning a Run
=============

``sage clone`` starts a new run from a stored checkpoint of another run. Use
it to change a run that has already started training, or to build on a
finished run.

.. code-block:: console

   $ sage clone --help
   usage: sage clone [-h] (--full-state | --weights-only) [--from-epoch FROM_EPOCH] old new

   Start a new run directory from a stored checkpoint of another run.

   positional arguments:
     old                   run directory to clone, or a directory inside it
     new                   new run directory

   options:
     -h, --help            show this help message and exit
     --full-state          weights, optimiser, schedule, EMA and mining bank; the epoch count
                           continues
     --weights-only        weights only; a new optimiser and schedule from epoch 0
     --from-epoch FROM_EPOCH
                           stored epoch to start from (default: the latest)

Two kinds of clone
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35
   :class: wrap

   * - Option
     - What the new run takes
     - Use it to
   * - ``--full-state``
     - The weights, optimiser, learning-rate schedule, averaged weights and
       mining bank. Training continues from the next epoch.
     - Carry on a run with changed settings.
   * - ``--weights-only``
     - The weights alone. Training starts again from epoch 0, with a new
       optimiser and schedule.
     - Build on a finished run: train again, starting from its network.

Making a clone
--------------

.. code-block:: console

   $ sage clone my_run my_run_more --full-state
   Cloned my_run epoch 1 into my_run_more (full state); training continues at epoch 2
   Settings: the same as that checkpoint's
   Next: sage check my_run_more

The new run directory holds:

``run.yaml`` and ``prior.yaml``
   Copies of the old run's, to change as you need.

``clone/``
   The checkpoint the new run starts from, and with ``--full-state`` the
   best one up to it.

``results/``
   With ``--full-state``, the losses and validation results up to that
   epoch.

``resolved/clone.yaml``
   Which run and checkpoint it came from.

``sage status`` shows where a clone came from:

.. code-block:: console

   $ sage status my_run_more
   my_run_more: training has not started
   Cloned from /path/to/my_run, epoch 1 (full state)
   Next: sage train my_run_more

The new run then trains like any other, with ``sage train my_run_more``.

Changing settings in a clone
----------------------------

Edit the new run's ``run.yaml`` and ``prior.yaml`` before training it. The
saved state limits which settings can change:

.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: wrap

   * - Settings
     - Can change
   * - ``model``, ``data.detectors``, ``signal.targets``, and the window:
       ``data.sample_rate_in_hz``, ``data.sample_length_in_s``,
       ``data.padding_length_in_s``
     - Never: the saved weights fit only these.
   * - ``optimiser``, ``schedule``, ``train.epochs``,
       ``train.samples_per_epoch``, ``train.batch_size``
     - With ``--weights-only`` only.
   * - Everything else
     - Always.

So to train a finished run for more epochs, clone it with
``--weights-only``. A change the saved state cannot take stops the clone's
training and names the setting:

.. code-block:: console

   $ sage train my_run_more
   error: run.yaml differs from the clone's start checkpoint (epoch 1) in settings its saved state cannot take: train.epochs: 2 -> 4
   fix: edit my_run_more/run.yaml back as above, then: sage train my_run_more

Changing a run that has started
-------------------------------

Once a run has saved its first checkpoint, its settings are fixed. Editing
its ``run.yaml`` stops the next ``sage train``, which points to
``sage clone``:

.. code-block:: console

   $ sage train my_run
   error: my_run: resolved/run.yaml differs after training started (validation.every_n_epochs: 1 -> 2); the changed settings continue in a clone of this run
   fix: sage clone my_run <new directory> --full-state

The clone takes the edited ``run.yaml`` and lists what changed. The old run
then trains on only with its own settings, so change its ``run.yaml`` back:

.. code-block:: console

   $ sage clone my_run my_run_val --full-state
   Cloned my_run epoch 1 into my_run_val (full state); training continues at epoch 2
   Settings that differ from that checkpoint:
     validation.every_n_epochs: 1 -> 2
   my_run trains on only with its recorded settings: revert these changes in my_run/run.yaml first
   Next: sage check my_run_val

Starting from an earlier epoch
------------------------------

A clone starts from the latest epoch unless ``--from-epoch`` names another
stored one:

.. code-block:: console

   $ sage clone my_run my_run_epoch0 --full-state --from-epoch 0
   Cloned my_run epoch 0 into my_run_epoch0 (full state); training continues at epoch 1
   Settings: the same as that checkpoint's
   Next: sage check my_run_epoch0

A run stores its last few epochs, as many as ``train.keep_checkpoints``.
