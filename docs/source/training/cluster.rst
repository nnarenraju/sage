Training on a Cluster
=====================

A full run trains for longer than most clusters let one job run. Sage
trains it in a chain of jobs: each job trains until near its time limit,
saves, and exits, and the next job resumes where it stopped.

The job script
--------------

Write a job script for your cluster, and keep it in the run directory. It
asks for a GPU and runs ``sage train`` with ``--stop-after`` set a little
under the job's time limit, so the job saves and exits before it is killed:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=my_run
   #SBATCH --partition=gpu
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=128G
   #SBATCH --time=2-00:00

   # Activate the environment Sage is installed in, if your cluster needs it
   cd /path/to/my_run
   sage train --stop-after 47h

Partition names, memory and accounts differ between clusters; ``sage
machine`` lists your cluster's partitions and limits. ``--cpus-per-task``
matches ``noise.read_workers``, the threads that read the strain.

Submitting a chain
------------------

``sage slurm-chain`` submits the script several times, each job starting
once the one before it has ended:

.. code-block:: console

   $ sage slurm-chain job.sbatch --jobs 3
   Submitted job <id> (1/3)
   Submitted job <id> (2/3, after <id>)
   Submitted job <id> (3/3, after <id>)
   Job output: /path/to/my_run/logs/jobs
   Next: squeue --me

Every job's output goes to ``logs/jobs/``, one file per job, whatever the
script asks for. A job that starts after training has finished ends at
once, so a chain can be longer than needed. ``--after`` makes the first job
wait for another, such as a job already running:

.. code-block:: console

   $ sage slurm-chain job.sbatch --jobs 3 --after <id>

``sage slurm-chain --help`` shows every option:

.. code-block:: console

   $ sage slurm-chain --help
   usage: sage slurm-chain [-h] --jobs JOBS [--after AFTER] script

   Submit your SLURM job script several times, each job after the previous one ends.

   positional arguments:
     script         your job script

   options:
     -h, --help     show this help message and exit
     --jobs JOBS    number of jobs in the chain
     --after AFTER  job id the first job waits for

Following a chain
-----------------

``sage status`` shows how far training has got across jobs, and estimates
when it will finish, not counting the time jobs wait in the queue.
``sage stop`` stops the chain: the job running stops after its current
epoch, and later jobs end without training. :doc:`follow` shows both.
