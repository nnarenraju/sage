The Data Directory
==================

Sage keeps everything it downloads and builds in one directory, the data
directory. Every run on the machine reads from it, so the data is downloaded
once and shared. ``sage setup`` chooses it, once per machine:

.. code-block:: console

   $ sage setup --data-dir /path/to/data
   Wrote /home/user/.config/sage/machine.yaml: data_dir /path/to/data
   Next: sage machine

``sage setup`` writes the machine file, ``~/.config/sage/machine.yaml``. Its
one setting is ``data_dir``:

.. code-block:: yaml

   # data_dir: the directory that holds all data Sage downloads and builds.
   data_dir: /path/to/data

To move the data directory later, run ``sage setup --data-dir`` again, or
edit the machine file.

``sage setup --help`` shows every option:

.. code-block:: console

   $ sage setup --help
   usage: sage setup [-h] [--data-dir DATA_DIR]

   Write the machine file: the directory for all data.

   options:
     -h, --help           show this help message and exit
     --data-dir DATA_DIR  data directory (default: fill it in yourself)

Without ``--data-dir``, ``sage setup`` writes the machine file with
``data_dir`` empty, for you to fill in:

.. code-block:: console

   $ sage setup
   Wrote /home/user/.config/sage/machine.yaml; open it and fill in data_dir
   Next: sage machine

The data directory must exist first. If it does not, ``sage setup`` stops
and gives the command that makes it:

.. code-block:: console

   $ sage setup --data-dir /path/to/data
   error: /path/to/data is not a directory
   fix: mkdir -p /path/to/data && sage setup --data-dir /path/to/data

Checking the machine
--------------------

``sage machine`` shows the machine file and the data directory, checks that
you can write to it, and names the job scheduler it finds:

.. code-block:: console

   $ sage machine
   Machine file: /home/user/.config/sage/machine.yaml
   data_dir: /path/to/data
   Scheduler: none (no sbatch or condor_submit on PATH); jobs run where they are started

On a SLURM cluster, it also lists the cluster's partitions and your account's
limits.

If ``data_dir`` is still empty, it says so:

.. code-block:: console

   $ sage machine
   error: data_dir in /home/user/.config/sage/machine.yaml is empty; fill it in
   fix: sage setup --data-dir <directory>

What the data directory holds
-----------------------------

``sage list data`` lists the data directory's folders. After the
:doc:`../quickstart`, it holds the strain of O3a and O3b and the ASDs built
from it:

.. code-block:: console

   $ sage list data
   Data in /path/to/data:
     asds: o3a, o3a+o3b, o3b
     o3a: search, training
     o3b: training

.. list-table::
   :header-rows: 1
   :widths: 35 65
   :class: wrap

   * - Folder
     - What it holds
   * - ``o3a/training/``
     - The strain of O3a that training reads. :doc:`download` explains it.
   * - ``o3a/search/``
     - The strain of O3a that a search reads. :doc:`../searching` explains
       it.
   * - ``asds/o3a/``
     - The ASDs built from O3a's strain. :doc:`asds` explains them.
   * - ``asds/o3a+o3b/``
     - The fiducial ASDs, built from O3a and O3b together.

Each observing run gets its own folders in the same way, ``o3b/`` here.
