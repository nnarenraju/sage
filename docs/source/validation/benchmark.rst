The MLGWSC-1 Benchmark
======================

``sage benchmark --mlgwsc1`` tests a trained network on MLGWSC-1, the first
Machine Learning Gravitational-Wave Search Mock Data Challenge, and compares
it with every team that took part. It uses the challenge's most realistic study,
dataset 4: IMRPhenomXPHM signals in a month of real O3a noise.

.. code-block:: console

   $ sage benchmark --mlgwsc1

It benchmarks ``best.pt`` unless ``--from-epoch`` or ``--ema-calibrated``
names another stored checkpoint, as in :doc:`validate`.

What it makes
-------------

The first benchmark on a machine makes, once, in ``mlgwsc1/`` in the data
directory:

* MLGWSC-1's own code, which makes the dataset and evaluates every search;
* every team's published results;
* dataset 4, and the signal-to-noise ratio of each of its injections.

Each benchmark then runs the network over dataset 4, and evaluates its
events with MLGWSC-1's code. The results go into the run directory:

.. code-block:: console

   $ sage benchmark --mlgwsc1
   Benchmarked on MLGWSC-1 dataset 4: events and evaluation in results/mlgwsc1/best
   Next: sage status

A benchmark that stops resumes when run again, keeping the parts already
finished. A checkpoint is benchmarked once: if ``results/mlgwsc1/best``
already exists, ``sage benchmark`` stops.

``sage compare`` draws a benchmarked run's results beside other runs':
:doc:`compare`.

``sage benchmark --help`` shows every option:

.. code-block:: console

   $ sage benchmark --help
   usage: sage benchmark [-h] [--mlgwsc1] [--from-epoch FROM_EPOCH] [--ema-calibrated]
                         [--workers WORKERS]
                         [directory]

   Benchmark one stored checkpoint on MLGWSC-1 dataset 4 against every team.

   positional arguments:
     directory             run directory, or a directory inside it (default: here)

   options:
     -h, --help            show this help message and exit
     --mlgwsc1             the MLGWSC-1 dataset 4 benchmark
     --from-epoch FROM_EPOCH
                           a stored epoch instead of best.pt
     --ema-calibrated      ema_calibrated.pt instead of best.pt
     --workers WORKERS     processes for the injection SNRs (default: 16)
