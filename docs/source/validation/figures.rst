Figures
=======

``sage plot`` draws the validation figures of a run into ``results/plots/``,
one folder for each kind of figure, about two dozen kinds in all:

.. code-block:: console

   $ sage plot
   Drew the figures for epoch_0127 in results/plots
   Next: sage help

By default it draws the best validated epoch. ``--group`` draws another
validation pass instead, such as an epoch, the final check's ``best.pt``
(``final_best``) or ``ema_calibrated.pt`` (``final_ema``):

.. code-block:: console

   $ sage plot --group final_best
   Drew the figures for final_best in results/plots
   Next: sage help

.. code-block:: console

   $ sage plot --help
   usage: sage plot [-h] [--group GROUP] [directory]

   Draw the run's validation figures into results/plots.

   positional arguments:
     directory      run directory, or a directory inside it (default: here)

   options:
     -h, --help     show this help message and exit
     --group GROUP  a validation group instead of the best validated epoch, e.g. epoch_0010,
                    final_best or validate_best

Examples
--------

These figures come from a run of the ``three_class`` preset, 128 epochs,
drawn with ``sage plot --group final_best``.

.. figure:: /_static/validation/loss_curves.png
   :width: 70%
   :align: center

   **Loss curves.** The training and validation loss of every epoch. The star
   marks the epoch with the lowest validation loss, saved as ``best.pt``.

.. figure:: /_static/validation/roc.png
   :width: 70%
   :align: center

   **ROC curve.** The fraction of signals found against the fraction of noise
   windows wrongly flagged, as the threshold on the network's output moves.

.. figure:: /_static/validation/raw_output.png
   :width: 100%
   :align: center

   **Network output.** Left: how often the network gives each value of its
   ranking statistic, for signals, noise and non-astrophysical windows.
   Right: the fraction of signals above each threshold.

.. figure:: /_static/validation/diagonal_mchirp.png
   :width: 70%
   :align: center

   **Chirp mass.** The network's estimate of the chirp mass against the true
   value. Along the dashed line, the estimate is exact.

.. figure:: /_static/validation/efficiency_masses.png
   :width: 70%
   :align: center

   **Efficiency by mass.** The fraction of signals the network scores above
   0.5, in each bin of the two component masses.
