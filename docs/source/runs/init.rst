Making a Run Directory
======================

``sage init`` makes a run directory from a preset:

.. code-block:: console

   $ sage init my_run --preset quick
   Made my_run: run.yaml, every setting of the quick preset, and prior.yaml, the bbh_narrow prior
   Next: cd my_run && sage download

Like most commands, ``sage init`` ends with a ``Next:`` line naming the
command to run next.

The directory's name is the run's name. It starts with two files:

``run.yaml``
   Every setting of the run, copied from the preset. :doc:`run_file`
   explains it.

``prior.yaml``
   The prior the training signals are drawn from, copied from a prior
   preset. :doc:`prior_file` explains it.

Both are yours to change.

Each run preset comes with a prior, and ``sage init`` copies it. To start
from another prior preset, name it with ``--prior``; ``sage list priors``
names them:

.. code-block:: console

   $ sage init my_run --preset quick --prior <name>

``sage init --help`` shows every option:

.. code-block:: console

   $ sage init --help
   usage: sage init [-h] [--preset PRESET] [--prior PRIOR] directory

   Make a run directory with every setting of a run preset.

   positional arguments:
     directory        run directory to make; its name is the run's name

   options:
     -h, --help       show this help message and exit
     --preset PRESET  run preset to start from (listed when not given)
     --prior PRIOR    prior preset to copy into prior.yaml (default: the run preset's)

Without ``--preset``, ``sage init`` lists the presets to choose from:

.. code-block:: console

   $ sage init my_run
   Run presets:
     broad
     mining
     quick
     three_class
     three_detector
     two_detector
   Next: sage init my_run --preset <name>

``sage init`` never overwrites a run. If the directory already holds a
``run.yaml``, it stops and suggests a new directory:

.. code-block:: console

   $ sage init my_run --preset quick
   error: my_run/run.yaml exists
   fix: sage init <new directory> --preset quick

Working in a run directory
--------------------------

Every command after ``sage init`` acts on one run.

From inside the run directory, or any folder inside it, give the command
alone:

.. code-block:: console

   $ cd my_run
   $ sage train

From outside, name the run directory:

.. code-block:: console

   $ sage train my_run

As the run trains, Sage adds folders for its checkpoints, results and logs
next to the two files. :doc:`../training` describes them.
