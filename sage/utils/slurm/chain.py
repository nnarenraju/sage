"""Chain a user's SLURM job script: each job starts once the previous one has ended."""

import subprocess


def chain(script, jobs: int, after: str | None = None, environment: dict | None = None) -> list:
    """
    Submit ``script`` ``jobs`` times, each depending (``afterany``) on the previous job.

    Parameters
    ----------
    script : Path
        The user's job script.
    jobs : int
        Jobs to submit.
    after : str or None
        Job the first one waits for.
    environment : dict or None
        Variables exported to the jobs on top of the submitting environment.

    Returns
    -------
    list of str
        Job ids, in order.

    Raises
    ------
    RuntimeError
        With sbatch's message and the jobs already submitted, if a submission fails.
    """
    ids, previous = [], after
    export = [] if environment is None else ["--export=" + ",".join(["ALL", *(f"{k}={v}" for k, v in environment.items())])]
    for _ in range(jobs):
        dependency = [] if previous is None else [f"--dependency=afterany:{previous}"]
        done = subprocess.run(["sbatch", "--parsable", *export, *dependency, str(script)], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"sbatch: {done.stderr.strip()} (already submitted: {ids or 'none'})")
        previous = done.stdout.strip().split(";")[0]  # "id;cluster" on federated sites
        ids.append(previous)
    return ids
