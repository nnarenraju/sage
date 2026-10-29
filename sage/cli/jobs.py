"""``sage chain``: submit a user's job script as a chain of jobs."""

import argparse
from pathlib import Path

from sage.cli.registry import CommandError, command
from sage.utils.schedulers import schedulers
from sage.utils.slurm.chain import chain


@command("chain", "Submit your job script several times, each job after the previous one ends")
def chain_jobs(arguments: list) -> None:
    """Submit ``script`` ``--jobs`` times as a dependency chain (SLURM ``afterany``)."""
    parser = argparse.ArgumentParser(
        prog="sage chain", description="Submit your job script several times, each job after the previous one ends."
    )
    parser.add_argument("script", help="your job script")
    parser.add_argument("--jobs", type=int, required=True, help="number of jobs in the chain")
    parser.add_argument("--after", help="job id the first job waits for")
    options = parser.parse_args(arguments)

    script = Path(options.script)
    if not script.is_file():
        raise CommandError(f"no job script {script}", f"sage chain <your job script> --jobs {options.jobs}")
    if options.jobs < 1:
        raise CommandError(f"--jobs {options.jobs}: a chain needs at least one job", f"sage chain {script} --jobs 1")
    found = schedulers()
    if found != ["slurm"]:
        raise CommandError(
            "no scheduler to chain on (no sbatch on PATH)" if not found
            else f"chaining on {', '.join(found)} is not available yet",
            "sage train",
        )
    try:
        ids = chain(script, options.jobs, options.after)
    except RuntimeError as error:
        raise CommandError(str(error), f"sage chain {script} --jobs {options.jobs}") from None
    for position, job in enumerate(ids, 1):
        waits = f", after {ids[position - 2]}" if position > 1 else (f", after {options.after}" if options.after else "")
        print(f"Submitted job {job} ({position}/{len(ids)}{waits})")
    print("Next: squeue --me")
