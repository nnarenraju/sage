Making a Run
============

A run is one network: a directory that holds the network's settings and,
once it trains, everything training writes. These pages show how to make a
run, what its settings mean and how to change them.

:doc:`presets`
   The ready-made runs that come with Sage, and which one to start from.

:doc:`init`
   Making a run directory with ``sage init``, and the files it holds.

:doc:`run_file`
   ``run.yaml``: every setting of the run, section by section.

:doc:`prior_file`
   ``prior.yaml``: the parameters the training signals are drawn from.

:doc:`changing`
   Worked examples of common changes: waveform, detectors, noise, network,
   length of training.

:doc:`cloning`
   Starting a new run from another run's checkpoint: to change a run that
   has started, or to build on a finished one.

:doc:`mistakes`
   What Sage prints when a setting is wrong, and how to fix it.

Every setting, with its meaning and allowed values, is also listed in
:doc:`../configuration`.

.. toctree::
   :hidden:

   presets
   init
   run_file
   prior_file
   changing
   cloning
   mistakes
