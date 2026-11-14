"""
What ``sage`` alone prints: the journey through the commands.
"""

from importlib.metadata import version

from sage.cli.registry import COMMANDS


def journey() -> str:
    """
    The journey text: the commands in order, then the others.

    Returns
    -------
    str
        Text printed by ``sage`` alone or ``sage help``.
    """
    steps = sorted((entry["journey"], name, entry["summary"]) for name, entry in COMMANDS.items()
                   if entry["journey"] is not None)
    width = max(len(name) for _, name, _ in steps)
    lines = [f"sage {version('sage-gw')}", "", "From a new machine to a trained network:"]
    lines += [f"  {position}. sage {name:<{width}}  {summary}" for position, (_, name, summary) in enumerate(steps, 1)]
    others = sorted((name, entry["summary"]) for name, entry in COMMANDS.items() if entry["journey"] is None)
    if others:
        width = max(len(name) for name, _ in others)
        lines += ["", "Other commands:"] + [f"  sage {name:<{width}}  {summary}" for name, summary in others]
    lines += ["", "Help on a command: sage <command> --help"]
    return "\n".join(lines)
