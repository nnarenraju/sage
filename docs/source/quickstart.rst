Quick Start
===========

This guide runs Sage from scratch in the shortest way: from a new machine to
a trained network and the triggers of a search of real observing-run data.
Every step is one or a few commands, and nothing needs changing on the way.

It uses a *preset*, a ready-made set of settings. The ``quick`` preset is the
smallest: a network for the H1 and L1 detectors, trained on O3b noise and
validated on O3a.

``sage`` alone lists every command, and ``sage <command> --help`` explains
one.

Set up the machine
------------------

.. code-block:: bash

   sage setup --data-dir /path/to/data

This tells Sage where to keep everything it downloads and builds. It is done
once per machine.

Make a run
----------

.. code-block:: bash

   sage init my_run --preset quick
   cd my_run

This makes the run directory ``my_run/`` with two files:

* ``run.yaml``: every setting of the run;
* ``prior.yaml``: the distribution the training signals are drawn from.

Both can be used as they are. To change the run, edit them now, before the
next steps.

Get the data
------------

.. code-block:: bash

   sage download
   sage build-asds

``sage download`` fetches from GWOSC the strain the run needs: O3a and O3b,
for H1 and L1. ``sage build-asds`` then builds from it the amplitude spectral
densities (ASDs) the run whitens with. Both read ``run.yaml``, and both resume if stopped.

Train
-----

.. code-block:: bash

   sage train
   sage plot

``sage train`` trains the network on a GPU and validates it as it goes. It
resumes if stopped, and ``sage status`` shows how far it has got. ``sage
plot`` then draws the validation figures into ``results/plots/``.

Search
------

To search all of O3a with the trained network:

.. code-block:: bash

   sage download o3a --search --detectors H1 L1
   sage search-init o3a --checkpoint best

``sage download ... --search`` fetches O3a for searching, with nothing cut
out. ``sage search-init`` makes the search ``search/o3a_best/``, using the
network of the best validated epoch, ``best.pt``.

In ``search/o3a_best/search.yaml``, set ``live_time_in_years: 0.5``, for six
months of background. Then:

.. code-block:: bash

   sage search o3a_best
   sage statmap o3a_best

``sage search`` scores every window of O3a where both detectors have data,
then the background: the same data with L1 shifted against H1, which tells
how often noise alone makes a trigger. It resumes if stopped.

``sage statmap`` gives every trigger its false-alarm rate, in
``search/o3a_best/statmap/``. It reads only finished scores, so it can run
while the search is still going.
