Training
========

With the data in place, a run can train. These pages show how to check a
run, train it and follow it, what training writes, and how to train on a
cluster.

:doc:`check`
   Checking a run with ``sage check`` before it trains.

:doc:`train`
   Training a run with ``sage train``: what each epoch does, the final
   check, and resuming.

:doc:`follow`
   Following training with ``sage status`` and ``sage log``, and stopping it
   with ``sage stop``.

:doc:`outputs`
   The checkpoints, results and logs a run writes.

:doc:`cluster`
   Training on a cluster: job scripts, time limits and ``sage slurm-chain``.

Looking at the results is on :doc:`../validation`.

.. toctree::
   :hidden:

   check
   train
   follow
   outputs
   cluster
