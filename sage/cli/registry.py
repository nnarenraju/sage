"""The command registry and the error every command raises."""

import difflib

# Command name -> {"function", "summary", "journey"}; filled as the command modules are imported.
COMMANDS = {}


class CommandError(Exception):
    """
    A command failed.

    Parameters
    ----------
    message : str
        What went wrong.
    fix : str
        The exact command that fixes it.
    """

    def __init__(self, message: str, fix: str):
        super().__init__(message)
        self.fix = fix


def closest(name: str, valid: list) -> str | None:
    """The accepted name closest to ``name`` (as ``loader.suggestion`` matches), or None."""
    close = difflib.get_close_matches(name, valid, n=1)
    return close[0] if close else None


def command(name: str, summary: str, journey: int | None = None):
    """
    Register ``function(arguments)`` as ``sage <name>``.

    Parameters
    ----------
    name : str
        Command name: one word, or words joined by single hyphens.
    summary : str
        One-line description.
    journey : int or None
        Position in the journey ``sage`` prints; None for commands outside it.
    """
    def register(function):
        COMMANDS[name] = {"function": function, "summary": summary, "journey": journey}
        return function

    return register
