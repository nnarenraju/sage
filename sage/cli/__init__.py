"""
The ``sage`` command line: ``sage <command>``, also ``sgw <command>``,
``sage-<command>`` and ``python -m sage <command>``.

Every command acts on the current directory unless given one and ends by
printing the next command; every error ends with the exact command that fixes it.
"""

import sys
from importlib.metadata import version
from pathlib import Path

from sage.cli import data as _data  # noqa: F401  (registers its commands)
from sage.cli import jobs as _jobs  # noqa: F401  (registers its commands)
from sage.cli import machine as _machine  # noqa: F401  (registers its commands)
from sage.cli import results as _results  # noqa: F401  (registers its commands)
from sage.cli import runs as _runs  # noqa: F401  (registers its commands)
from sage.cli import training as _training  # noqa: F401  (registers its commands)
from sage.cli.journey import journey
from sage.cli.registry import COMMANDS, CommandError, closest
from sage.config.loader import suggestion


def main(argv: list | None = None) -> int:
    """
    Run one command.

    Parameters
    ----------
    argv : list of str or None
        Command and its arguments; None reads ``sys.argv``, where a
        ``sage-<command>`` executable names the command itself.

    Returns
    -------
    int
        Exit status: 0, or 1 after an error.
    """
    if argv is None:
        argv = sys.argv[1:]
        program = Path(sys.argv[0]).name
        if program.startswith("sage-"):
            argv = [program.removeprefix("sage-"), *argv]
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(journey())
        return 0
    if argv[0] == "--version":
        print(f"sage {version('sage-gw')}")
        return 0

    name, arguments = argv[0], argv[1:]
    try:
        if name not in COMMANDS:
            names = sorted([*COMMANDS, "help"])
            match = closest(name, names)
            fix = " ".join(["sage", match, *arguments]) if match else "sage help"
            raise CommandError(f"unknown command {name!r}; {suggestion(name, names)}", fix)
        COMMANDS[name]["function"](arguments)
    except CommandError as error:
        print(f"error: {error}\nfix: {error.fix}", file=sys.stderr)
        return 1
    return 0
