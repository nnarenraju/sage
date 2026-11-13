"""
Chain a user's SLURM job script: each job starts once the previous one
has ended.
"""

import subprocess
from pathlib import Path

# Under the run directory; one <job name>-<job id>.out per job
JOB_LOGS = Path("logs") / "jobs"


def chain(script, run_dir: Path, jobs: int, after: str | None = None) -> list:
    """
    Submit ``script`` ``jobs`` times, each after the previous job ends.

    Each job depends (``afterany``) on the previous one. Its output and
    errors go to ``<run_dir>/logs/jobs/<job name>-<job id>.out``,
    whatever the script asks for.

    Parameters
    ----------
    script : str or Path
        The user's job script.
    run_dir : Path
        The run directory the jobs work on.
    jobs : int
        Jobs to submit.
    after : str or None
        Job the first one waits for.

    Returns
    -------
    list of str
        Job ids, in order.

    Raises
    ------
    RuntimeError
        If a submission fails; the message gives sbatch's error and the
        jobs already submitted.
    OSError
        If ``sbatch`` cannot be run.
    """
    logs = Path(run_dir) / JOB_LOGS
    logs.mkdir(parents=True, exist_ok=True)  # SLURM does not create it
    path = logs.resolve() / "%x-%j.out"
    output = [f"--output={path}", f"--error={path}"]  # stdout and stderr in one file
    ids, previous = [], after
    for _ in range(jobs):
        dependency = [] if previous is None else [f"--dependency=afterany:{previous}"]
        done = subprocess.run(
            ["sbatch", "--parsable", *output, *dependency, str(script)], capture_output=True, text=True
        )
        if done.returncode != 0:
            raise RuntimeError(f"sbatch: {done.stderr.strip()} (already submitted: {ids or 'none'})")
        previous = done.stdout.strip().split(";")[0]  # "id;cluster" on federated sites
        ids.append(previous)
    return ids
