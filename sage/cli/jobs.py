"""``sage slurm-chain``: submit a user's SLURM job script as a chain of jobs."""

import argparse
from pathlib import Path

from sage.cli.registry import CommandError, command
from sage.utils.schedulers import schedulers
from sage.utils.slurm.chain import JOB_LOGS, chain


@command("slurm-chain", "Submit your SLURM job script several times, each job after the previous one ends")
def chain_jobs(arguments: list) -> None:
    """Submit ``script`` ``--jobs`` times as a dependency chain (SLURM ``afterany``)."""
    parser = argparse.ArgumentParser(
        prog="sage slurm-chain",
        description="Submit your SLURM job script several times, each job after the previous one ends."
    )
    parser.add_argument("script", help="your job script")
    parser.add_argument("--jobs", type=int, required=True, help="number of jobs in the chain")
    parser.add_argument("--after", help="job id the first job waits for")
    options = parser.parse_args(arguments)

    script = Path(options.script)
    if not script.is_file():
        raise CommandError(f"no job script {script}", f"sage slurm-chain <your job script> --jobs {options.jobs}")
    run_dir = script.resolve().parent  # job output goes to its logs/jobs
    if not (run_dir / "run.yaml").exists():
        raise CommandError(f"{script}: a job script lives in the run directory it trains, and {run_dir} has no "
                           "run.yaml", f"mv {script} <run directory>/, then: sage slurm-chain <run directory>/{script.name} "
                           f"--jobs {options.jobs}")
    if options.jobs < 1:
        raise CommandError(f"--jobs {options.jobs}: a chain needs at least one job", f"sage slurm-chain {script} --jobs 1")
    if "slurm" not in schedulers():
        raise CommandError("no SLURM on this machine (no sbatch on PATH)", "sage train")
    try:
        ids = chain(script, run_dir, options.jobs, options.after)
    except RuntimeError as error:
        raise CommandError(str(error), f"sage slurm-chain {script} --jobs {options.jobs}") from None
    for position, job in enumerate(ids, 1):
        waits = f", after {ids[position - 2]}" if position > 1 else (f", after {options.after}" if options.after else "")
        print(f"Submitted job {job} ({position}/{len(ids)}{waits})")
    print(f"Job output: {run_dir / JOB_LOGS}")
    print("Next: squeue --me")
