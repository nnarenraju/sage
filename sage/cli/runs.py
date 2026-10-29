"""``sage init``, ``sage copy-prior``, ``sage show`` and ``sage list``: making and reading a run directory."""

import argparse
import inspect
import shutil
from pathlib import Path

import yaml

from sage.cli.journey import next_command
from sage.cli.registry import CommandError, closest, command
from sage.config import machine
from sage.config import loader
from sage.config.loader import load_run, suggestion


def parameters(cls, provided: set, overrides: dict | None = None) -> str:
    """Each parameter a run file may set for ``cls``, with its default, or marked required."""
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
    """``(name, parameters)`` of every model, loss, waveform, optimiser or schedule."""
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
    """What is downloaded under the machine's data directory."""
    try:
        data_dir = machine.read_data_dir()
    except (FileNotFoundError, ValueError) as error:
        raise CommandError(str(error), "sage machine-setup") from None
    lines = [f"Data in {data_dir}:"]
    for entry in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        parts = sorted(p.name for p in entry.iterdir() if p.is_dir())
        lines.append(f"  {entry.name}: {', '.join(parts) or 'empty'}")
    return lines if len(lines) > 1 else lines + ["  nothing yet"]


THINGS = ("presets", "priors", "models", "losses", "waveforms", "optimisers", "schedules", "asds", "data")


@command("list", "List presets, priors, models, losses, waveforms, optimisers, schedules, ASDs or data")
def list_things(arguments: list) -> None:
    """Print the available presets, components, analytic ASDs or downloaded data."""
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


@command("init", "Make a run directory from a run preset", journey=2)
def init_run(arguments: list) -> None:
    """Make a run directory whose ``run.yaml`` extends a run preset; without a preset, list them."""
    parser = argparse.ArgumentParser(prog="sage init", description="Make a run directory whose run.yaml extends a run preset.")
    parser.add_argument("directory", help="run directory to make; its name is the run's name")
    parser.add_argument("--preset", help="run preset to extend (listed when not given)")
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
    run_file = directory / "run.yaml"
    if run_file.exists():
        raise CommandError(f"{run_file} exists", f"sage show {directory}")

    directory.mkdir(parents=True, exist_ok=True)
    run_file.write_text(f"# Settings not listed here come from the preset; `sage show` prints them all.\n"
                        f"extends: {options.preset}\n")
    print(f"Made {directory} (run.yaml extends {options.preset})")
    print(f"Next: cd {directory} && sage show")


@command("copy-prior", "Copy a prior preset into the run directory to edit it")
def copy_prior(arguments: list) -> None:
    """Write a prior preset to ``prior.yaml`` in the run directory, unchanged, for editing."""
    parser = argparse.ArgumentParser(prog="sage copy-prior", description="Copy a prior preset into the run directory to edit it.")
    parser.add_argument("preset", help="prior preset")
    parser.add_argument("directory", nargs="?", default=".", help="run directory (default: here)")
    options = parser.parse_args(arguments)

    names = loader.preset_names("priors")
    if options.preset not in names:
        match = closest(options.preset, names)
        raise CommandError(f"unknown prior preset {options.preset!r}; {suggestion(options.preset, names)}",
                           f"sage copy-prior {match} {options.directory}" if match else "sage list priors")
    target = Path(options.directory) / "prior.yaml"
    if target.exists():
        raise CommandError(f"{target} exists", f"rm {target} && sage copy-prior {options.preset} {options.directory}")
    shutil.copyfile(loader.PRESETS / "priors" / f"{options.preset}.yaml", target)
    print(f"Wrote {target}, a copy of the {options.preset} prior preset. To use it, set in run.yaml:")
    print("  signal:\n    prior: prior.yaml")
    print("Next: sage show")


@command("show", "Print the run's settings with every default filled in")
def show_run(arguments: list) -> None:
    """Print ``run.yaml`` resolved: the preset merged and every default filled in; the prior is checked too."""
    from sage.config.prior import load_prior
    from sage.factory.build import resolve_settings

    parser = argparse.ArgumentParser(prog="sage show", description="Print the run's settings with every default filled in.")
    parser.add_argument("directory", nargs="?", default=".", help="run directory (default: here)")
    options = parser.parse_args(arguments)

    directory = Path(options.directory)
    run_file = directory / "run.yaml"
    if not run_file.exists():
        raise CommandError(f"no run.yaml in {directory}", "sage init <directory> --preset <name>")
    try:
        settings = resolve_settings(load_run(run_file))
        prior = load_prior(settings["signal"]["prior"], directory)
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        raise CommandError(str(error), f"edit {run_file} as above, then: sage show") from None
    print(yaml.safe_dump(settings, sort_keys=False).rstrip())
    print(f"# prior {settings['signal']['prior']}: {len(prior['priors'])} distributions")
    print(f"Next: {next_command(directory)}")


@command("clone", "Start a new run directory from a stored checkpoint of another run")
def clone_run(arguments: list) -> None:
    """Copy a run's ``run.yaml`` and one stored checkpoint into a new run directory, with its full state or weights only."""
    from sage.config.run_dir import StateMismatch
    from sage.factory.clone import EpochUnavailable, clone_run as clone
    from sage.factory.train_run import check_run

    parser = argparse.ArgumentParser(prog="sage clone", description="Start a new run directory from a stored checkpoint "
                                     "of another run.")
    parser.add_argument("old", help="run directory to clone")
    parser.add_argument("new", help="new run directory")
    state = parser.add_mutually_exclusive_group(required=True)
    state.add_argument("--full-state", action="store_true",
                       help="weights, optimiser, schedule, EMA and mining bank; the epoch count continues")
    state.add_argument("--weights-only", action="store_true", help="weights only; a new optimiser and schedule from epoch 0")
    parser.add_argument("--from-epoch", type=int, help="stored epoch to start from (default: the latest)")
    options = parser.parse_args(arguments)

    old, new = Path(options.old), Path(options.new)
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
