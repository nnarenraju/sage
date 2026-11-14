Presets
=======

A preset is a complete set of run settings that comes with Sage. Every run
starts from one: ``sage init`` copies a preset into the new run directory,
and you change the copy, never the preset.

``sage list presets`` names them:

.. code-block:: console

   $ sage list presets
   Presets:
     broad
     mining
     quick
     three_class
     three_detector
     two_detector

.. list-table::
   :header-rows: 1
   :widths: 22 18 60
   :class: wrap

   * - Preset
     - Detectors
     - What it trains
   * - ``quick``
     - H1, L1
     - A short run, 6 epochs of 6,400 samples, for a first try or to check
       a setup.
   * - ``two_detector``
     - H1, L1
     - A full run, 128 epochs of 2,000,000 samples.
   * - ``three_detector``
     - H1, L1, V1
     - ``two_detector``, with Virgo.
   * - ``mining``
     - H1, L1
     - ``two_detector``, with hard-noise mining.
   * - ``three_class``
     - H1, L1
     - ``mining``, with non-astrophysical noise as a third class.
   * - ``broad``
     - H1, L1
     - ``two_detector``, with the ``bbh_broad`` prior in 20 s windows.

All six train on O3b noise and validate on O3a, with binary black hole
signals. ``broad`` cannot train yet; its build is under construction.

Start with ``quick`` to check that everything works, then move to a full
run.

You cannot add presets of your own. To reuse a run's settings, copy its
``run.yaml`` and ``prior.yaml`` into a new directory. To build on a trained
run, clone it: :doc:`cloning` explains how.

Prior presets
-------------

The training signals' parameters come from a prior, and Sage has two prior
presets. ``sage list priors`` names them:

.. code-block:: console

   $ sage list priors
   Priors:
     bbh_broad
     bbh_narrow

.. list-table::
   :header-rows: 1
   :widths: 22 28 50
   :class: wrap

   * - Prior
     - Component masses
     - Used by
   * - ``bbh_narrow``
     - 7 to 50 solar masses
     - every run preset but ``broad``
   * - ``bbh_broad``
     - 5 to 100 solar masses
     - ``broad``

``sage init`` copies the preset's prior into the run directory as
``prior.yaml``, or another prior named with ``--prior``. :doc:`prior_file`
explains the file.
