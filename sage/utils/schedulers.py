"""Which job schedulers this machine has, from their client commands on PATH."""

import shutil

CLIENTS = {"slurm": "sbatch", "htcondor": "condor_submit"}


def schedulers() -> list:
    """Schedulers whose submit command is on PATH."""
    return [name for name, client in CLIENTS.items() if shutil.which(client)]
