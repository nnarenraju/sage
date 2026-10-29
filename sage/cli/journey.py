"""What ``sage`` alone prints: the journey through the commands, and the next one to run."""

from importlib.metadata import version
from pathlib import Path

from sage.cli.registry import COMMANDS
from sage.config import machine


def next_command(directory: Path = Path(".")) -> str:
    """The command to run next, from the state of the machine and of ``directory``."""
    if not machine.MACHINE_FILE.exists():
        return "sage machine-setup"
    if not (Path(directory) / "run.yaml").exists():
        return "sage init <directory> --preset <name>"
    if not (Path(directory) / "results" / "final_check.json").exists():
        return "sage train"
    return "sage plot"


def journey() -> str:
    """The journey (registered commands in order), the other commands and the next one to run."""
    steps = sorted((entry["journey"], name, entry["summary"]) for name, entry in COMMANDS.items()
                   if entry["journey"] is not None)
    width = max(len(name) for _, name, _ in steps)
    lines = [f"sage {version('sage-gw')}", "", "From a new machine to a trained network:"]
    lines += [f"  {position}. sage {name:<{width}}  {summary}" for position, (_, name, summary) in enumerate(steps, 1)]
    others = sorted((name, entry["summary"]) for name, entry in COMMANDS.items() if entry["journey"] is None)
    if others:
        width = max(len(name) for name, _ in others)
        lines += ["", "Other commands:"] + [f"  sage {name:<{width}}  {summary}" for name, summary in others]
    lines += ["", "Help on a command: sage <command> --help", f"Next: {next_command()}"]
    return "\n".join(lines)
