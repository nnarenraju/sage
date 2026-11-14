"""
``sage init``, ``list`` and ``clone``: making and
reading run directories.
"""

import argparse
import inspect
import shutil
from pathlib import Path

import yaml

from sage.cli.registry import CommandError, closest, command
from sage.config import machine
from sage.config import loader
from sage.config.loader import suggestion
from sage.config.prior import PRIOR_FILE
from sage.config.run_dir import find_run_dir


def parameters(cls, provided: set, overrides: dict | None = None) -> str:
    """
    Each parameter a run file may set for ``cls``, with its default.

    Parameters
    ----------
    cls : type
        Component class.
    provided : set of str
        Parameters Sage passes itself, left out.
    overrides : dict or None
        Defaults Sage uses instead of the signature's.

    Returns
    -------
    str
        ``name=default`` or ``name (required)`` per parameter, or
        ``no parameters``.
    """
    overrides = overrides or {}
    parts = []
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        if name == "self" or name in provided or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in overrides:
            parts.append(f"{name}={overrides[name]!r}")
        elif parameter.default is inspect.Parameter.empty:
            parts.append(f"{name} (required)")
        else:
            parts.append(f"{name}={parameter.default!r}")
    return ", ".join(parts) or "no parameters"


def components(thing: str) -> list:
    """
    Every model, loss, waveform, optimiser or schedule, with parameters.

    Parameters
    ----------
    thing : str
        ``models``, ``losses``, ``waveforms``, ``optimisers`` or
        ``schedules``.

    Returns
    -------
    list of tuple
        ``(name, parameters text)`` per component.
    """
    from sage.factory import build

    if thing == "schedules":
        lines = []
        for name, cls in build.SCHEDULES.items():
            if cls is build.WarmupPolynomialDecayFlatTail:
                lines.append((name, parameters(cls, build.SAGE_SCHEDULE_PROVIDED)))
            else:
                step = f"step_every (required: {', '.join(build.STEP_EVERY)})"
                lines.append((name, f"{step}, {parameters(cls, build.TORCH_SCHEDULE_PROVIDED)}"))
        return lines
    table, provided, overrides = {
        "models": (build.MODELS, build.MODEL_PROVIDED, {}),
        "losses": (build.LOSSES, build.LOSS_PROVIDED, {}),
        "waveforms": (build.WAVEFORMS, build.WAVEFORM_PROVIDED, {}),
        "optimisers": (build.OPTIMISERS, build.OPTIMISER_PROVIDED, build.OPTIMISER_DEFAULTS),
    }[thing]
    return [(name, parameters(cls, provided, overrides.get(name))) for name, cls in table.items()]


def data_lines() -> list:
    """
    What is downloaded under the machine's data directory.

    Returns
    -------
    list of str
        Report lines: each top-level directory and its subdirectories.

    Raises
    ------
    CommandError
        If the machine file is missing or invalid.
    """
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage setup --data-dir <directory>") from None
    lines = [f"Data in {data_dir}:"]
    for entry in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        parts = sorted(p.name for p in entry.iterdir() if p.is_dir())
        lines.append(f"  {entry.name}: {', '.join(parts) or 'empty'}")
    return lines if len(lines) > 1 else lines + ["  nothing yet"]


THINGS = ("presets", "priors", "models", "losses", "waveforms", "optimisers", "schedules", "asds", "data")


@command("list", "List presets, priors, models, losses, waveforms, optimisers, schedules, ASDs or data")
def list_things(arguments: list) -> None:
    """
    Print the presets, components, analytic ASDs or downloaded data.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage list``.

    Raises
    ------
    CommandError
        If the thing to list is unknown.
    """
    parser = argparse.ArgumentParser(prog="sage list", description="List what a run file can name, or what is downloaded.")
    parser.add_argument("thing", help=", ".join(THINGS))
    options = parser.parse_args(arguments)
    if options.thing not in THINGS:
        match = closest(options.thing, list(THINGS))
        raise CommandError(f"cannot list {options.thing!r}; {suggestion(options.thing, list(THINGS))}",
                           f"sage list {match or 'presets'}")

    if options.thing in ("presets", "priors"):
        names = loader.preset_names("runs" if options.thing == "presets" else "priors")
        lines = [f"{options.thing.capitalize()}:"] + [f"  {name}" for name in names or ["none yet"]]
    elif options.thing == "asds":
        from sage.data.noise.white_noise import available_asds

        lines = ["Analytic ASDs (synthetic noise):"] + [f"  {name}" for name in available_asds()]
    elif options.thing == "data":
        lines = data_lines()
    else:
        lines = [f"{options.thing.capitalize()} (parameters with their defaults):"]
        lines += [f"  {name}: {text}" for name, text in components(options.thing)]
    print("\n".join(lines))


@command("init", "Make a run directory with every setting of a run preset", journey=2)
def init_run(arguments: list) -> None:
    """
    Make a run directory holding a run preset's settings and a prior.

    ``run.yaml`` gets every setting, each component's parameters
    included, and ``prior.yaml`` a copy of the ``--prior`` preset, by
    default the run preset's. Without ``--preset``, the run presets are
    listed instead.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage init``.

    Raises
    ------
    CommandError
        If there are no presets, a preset is unknown or ``run.yaml`` or
        ``prior.yaml`` exists.
    """
    from sage.factory.build import resolve_settings

    parser = argparse.ArgumentParser(
        prog="sage init",
        description="Make a run directory with every setting of a run preset.",
    )
    parser.add_argument("directory", help="run directory to make; its name is the run's name")
    parser.add_argument(
        "--preset", help="run preset to start from (listed when not given)"
    )
    parser.add_argument(
        "--prior", help="prior preset to copy into prior.yaml (default: the run preset's)"
    )
    options = parser.parse_args(arguments)

    directory, names = Path(options.directory), loader.preset_names("runs")
    if options.preset is None:
        print("Run presets:\n" + "\n".join(f"  {name}" for name in names or ["none yet"]))
        print(f"Next: sage init {directory} --preset <name>")
        return
    if not names:
        raise CommandError("there are no run presets yet", "sage list presets")
    if options.preset not in names:
        match = closest(options.preset, names)
        raise CommandError(f"unknown run preset {options.preset!r}; {suggestion(options.preset, names)}",
                           f"sage init {directory} --preset {match}" if match else "sage list presets")
    priors = loader.preset_names("priors")
    if options.prior is not None and options.prior not in priors:
        match = closest(options.prior, priors)
        fix = f"sage init {directory} --preset {options.preset} --prior {match}" if match else "sage list priors"
        raise CommandError(f"unknown prior preset {options.prior!r}; {suggestion(options.prior, priors)}", fix)
    run_file, prior_file = directory / "run.yaml", directory / PRIOR_FILE
    for path in (run_file, prior_file):
        if path.exists():
            fix = f"sage init <new directory> --preset {options.preset}"
            raise CommandError(f"{path} exists", fix)

    preset = loader.PRESETS / "runs" / f"{options.preset}.yaml"
    settings = resolve_settings(loader.load_run(preset))
    # The run reads prior.yaml, so run.yaml names no prior
    signal = {key: value for key, value in settings["signal"].items() if key != "prior"}
    prior = options.prior or settings["signal"]["prior"]
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(loader.PRESETS / "priors" / f"{prior}.yaml", prior_file)
    header = f"# Made from the {options.preset} preset by sage init.\n\n"
    run_file.write_text(header + loader.yaml_text({**settings, "signal": signal}))
    print(f"Made {directory}: run.yaml, every setting of the {options.preset} preset, "
          f"and prior.yaml, the {prior} prior")
    print(f"Next: cd {directory} && sage download")


@command("clone", "Start a new run directory from a stored checkpoint of another run")
def clone_run(arguments: list) -> None:
    """
    Start a new run directory from a stored checkpoint of another run.

    Copies ``run.yaml`` and the checkpoint, with its full state or the
    weights only.

    Parameters
    ----------
    arguments : list of str
        Command-line arguments after ``sage clone``.

    Raises
    ------
    CommandError
        If the old run is invalid or has no such checkpoint, the new
        directory is not empty or the state cannot take the settings.
    """
    from sage.config.run_dir import StateMismatch
    from sage.factory.clone import EpochUnavailable, clone_run as clone
    from sage.factory.train_run import check_run

    parser = argparse.ArgumentParser(prog="sage clone", description="Start a new run directory from a stored checkpoint "
                                     "of another run.")
    parser.add_argument(
        "old", help="run directory to clone, or a directory inside it"
    )
    parser.add_argument("new", help="new run directory")
    state = parser.add_mutually_exclusive_group(required=True)
    state.add_argument("--full-state", action="store_true",
                       help="weights, optimiser, schedule, EMA and mining bank; the epoch count continues")
    state.add_argument("--weights-only", action="store_true", help="weights only; a new optimiser and schedule from epoch 0")
    parser.add_argument("--from-epoch", type=int, help="stored epoch to start from (default: the latest)")
    options = parser.parse_args(arguments)

    old, new = find_run_dir(options.old), Path(options.new)
    flag = "--full-state" if options.full_state else "--weights-only"
    latest = f"sage clone {old} {new} {flag}"
    again = latest + ("" if options.from_epoch is None else f" --from-epoch {options.from_epoch}")
    if not (old / "run.yaml").exists():
        raise CommandError(f"no run.yaml in {old}", f"sage clone <run directory> {new} {flag}")
    try:
        settings = check_run(old)[0]
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        raise CommandError(str(error), f"edit {old / 'run.yaml'} as above, then: {again}") from None
    try:
        epoch, changes = clone(old, new, settings, options.full_state, options.from_epoch)
    except StateMismatch as error:
        fix = again.replace(flag, "--weights-only") if error.weights_only else f"edit {old / 'run.yaml'} back as above, then: {again}"
        raise CommandError(str(error), fix) from None
    except EpochUnavailable as error:
        raise CommandError(str(error), latest) from None
    except FileExistsError as error:
        raise CommandError(str(error), f"sage clone {old} <new directory> {flag}") from None
    except FileNotFoundError as error:
        raise CommandError(str(error), f"sage train {old}") from None
    except ValueError as error:
        raise CommandError(str(error), f"sage clone {old} {new} --weights-only") from None

    start = f"training continues at epoch {epoch + 1}" if options.full_state else "training starts at epoch 0"
    print(f"Cloned {old} epoch {epoch} into {new} ({'full state' if options.full_state else 'weights only'}); {start}")
    print("Settings that differ from that checkpoint:\n" + "\n".join(f"  {change}" for change in changes)
          if changes else "Settings: the same as that checkpoint's")
    if changes:
        print(f"{old} trains on only with its recorded settings: revert these changes in {old / 'run.yaml'} first")
    print(f"Next: sage check {new}")
