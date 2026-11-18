Comparing Runs
==============

``sage compare`` scores the ``best.pt`` of several runs on the same windows,
so their results can be compared directly:

.. code-block:: console

   $ sage compare my_mining_run my_three_class_run --output compare_runs

The first run's validation settings define the windows: by default ten
million noise-only windows and one million with a signal. The runs must
analyse windows of the same shape, with the same detectors and window
length. ``--output`` names a new directory for the results.

What it writes
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - File
     - What it holds
   * - ``report.txt``
     - The comparison in text, also printed when ``sage compare`` ends.
   * - ``compare.h5``
     - Every run's outputs on every window.
   * - ``*.png``
     - Figures of the runs' outputs and estimates, side by side.

``report.txt`` gives, for each run:

* the fraction of signals found at a fixed false-alarm probability per noise
  window, from one in a thousand to one in ten million;
* that fraction in bins of signal-to-noise ratio;
* the loudest noise windows.

A run benchmarked on MLGWSC-1 also gets the benchmark's figures, side by
side with the others: :doc:`benchmark` explains the benchmark.

``sage compare --help`` shows every option:

.. code-block:: console

   $ sage compare --help
   usage: sage compare [-h] --output OUTPUT [--noise-windows NOISE_WINDOWS]
                       [--signal-windows SIGNAL_WINDOWS] [--seed SEED]
                       runs [runs ...]

   Score several runs' best.pt on identical validation windows.

   positional arguments:
     runs                  run directories, or directories inside them; the first one's validation
                           settings define the windows

   options:
     -h, --help            show this help message and exit
     --output OUTPUT       new directory for compare.h5, report.txt and the figures
     --noise-windows NOISE_WINDOWS
                           noise-only windows (default: 10000000)
     --signal-windows SIGNAL_WINDOWS
                           windows with a signal (default: 1000000)
     --seed SEED           seed of the windows (default: 170817)
