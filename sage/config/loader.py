"""Read a run file: one level of ``extends``, values typed by the schema.

Plain YAML scalars are read as strings and converted by the type each
setting declares, so ``2e-4`` is a number and ``yes`` is never a bool.
"""

import collections.abc
import difflib
import inspect
import types
import typing
from pathlib import Path
from typing import Any

import torch
import yaml

from sage.config.schema import NULL, SECTIONS, OneOf

PRESETS = Path(__file__).resolve().parents[1] / "presets"


class _StringLoader(yaml.SafeLoader):
    """Safe YAML loader that reads every plain scalar as a string."""


_StringLoader.yaml_implicit_resolvers = {}


def read_yaml(path: Path) -> dict:
    """
    Read a YAML file whose top level is a mapping.

    Parameters
    ----------
    path : Path
        YAML file.

    Returns
    -------
    dict
        Contents, with every scalar as a string.

    Raises
    ------
    ValueError
        If the top level is not a mapping.
    """
    with open(path) as file:
        return read_yaml_text(file.read(), str(path))


def read_yaml_text(text: str, where: str) -> dict:
    """
    Read YAML text whose top level is a mapping, every scalar as a string.

    Parameters
    ----------
    text : str
        YAML text (a file's contents, or a record a checkpoint carries).
    where : str
        Source, for error messages.

    Returns
    -------
    dict
        Contents.

    Raises
    ------
    ValueError
        If the top level is not a mapping.
    """
    values = yaml.load(text, Loader=_StringLoader)
    if not isinstance(values, dict):
        raise ValueError(f"{where}: expected a mapping at the top level")
    return values


def merge(base: dict, override: dict) -> dict:
    """
    Merge two mappings key by key; any other value is replaced whole.

    Parameters
    ----------
    base, override : dict
        Mappings; ``override`` wins.

    Returns
    -------
    dict
        Merged mapping.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def suggestion(name: str, valid: list) -> str:
    """
    Did-you-mean text for an unknown name.

    Parameters
    ----------
    name : str
        Unknown name.
    valid : list of str
        Accepted names.

    Returns
    -------
    str
        The closest accepted name, or the full list if none is close.
    """
    close = difflib.get_close_matches(name, valid, n=1)
    if close:
        return f"did you mean {close[0]!r}?"
    return f"valid: {', '.join(valid)}"


def preset_names(kind: str) -> list:
    """Names of the ``"runs"`` or ``"priors"`` presets."""
    return sorted(p.stem for p in (PRESETS / kind).glob("*.yaml"))


def read_preset(kind: str, name: str) -> dict:
    """
    Read a preset, which may not extend another preset.

    Parameters
    ----------
    kind : str
        ``"runs"`` or ``"priors"``.
    name : str
        Preset name.

    Returns
    -------
    dict
        Preset contents, not yet checked.

    Raises
    ------
    ValueError
        If the preset is unknown or itself extends another preset.
    """
    label = kind.removesuffix("s")
    presets = preset_names(kind)
    if name not in presets:
        raise ValueError(f"unknown {label} preset {name!r}; {suggestion(name, presets)}")
    values = read_yaml(PRESETS / kind / f"{name}.yaml")
    if "extends" in values:
        raise ValueError(f"{label} preset {name!r} extends another preset")
    return values


def read_extending(path: Path, kind: str) -> dict:
    """
    Read a file and merge it onto the preset it extends, if any.

    Parameters
    ----------
    path : Path
        Run or prior file.
    kind : str
        ``"runs"`` or ``"priors"``.

    Returns
    -------
    dict
        Merged contents, not yet checked.
    """
    values = read_yaml(path)
    name = values.pop("extends", None)
    if name is None:
        return values
    return merge(read_preset(kind, name), values)


def spec_from_hint(hint: Any, where: str) -> Any:
    """
    The spec of a type hint, with typing forms reduced to what YAML can hold.

    ``Optional[X]`` is ``X`` or ``null``; a ``Union`` drops ``torch.Tensor``
    and members with no YAML form, and becomes a :class:`OneOf` if several
    remain; ``Literal`` gives its choices; ``list[X]``, ``Iterable[X]`` and
    ``tuple[X, ...]`` are lists of ``X``.

    Parameters
    ----------
    hint : Any
        Type hint of an ``__init__`` argument.
    where : str
        Dotted location, for error messages.

    Returns
    -------
    Any
        Spec for :func:`convert`.

    Raises
    ------
    ValueError
        If the hint has no YAML form (a function, an optimiser, ...).
    """
    if hint in (bool, int, float, str):
        return hint
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Literal:
        return tuple(args)
    if origin in (typing.Union, types.UnionType):
        members, refusal = [], None
        for arg in args:
            if arg in (type(None), torch.Tensor):
                continue
            try:
                members.append(spec_from_hint(arg, where))
            except ValueError as error:
                refusal = error
        if type(None) in args:
            members.append(NULL)
        if not members:
            raise refusal
        return members[0] if len(members) == 1 else OneOf(*members)
    if origin in (list, tuple, collections.abc.Iterable):
        return [spec_from_hint(args[0], where)]
    raise ValueError(f"{where}: takes {hint}, which cannot be written in a run file")


def convert(value: Any, spec: Any, where: str) -> Any:
    """
    Convert one value to the type its spec declares.

    Parameters
    ----------
    value : Any
        Value as read from YAML.
    spec : type, tuple, list, dict or OneOf
        Declared type, allowed strings, list of a spec, nested schema, or
        alternatives tried in order.
    where : str
        Dotted location, for error messages.

    Returns
    -------
    Any
        Converted value.

    Raises
    ------
    ValueError
        If the value does not match the spec.
    """
    if isinstance(spec, OneOf):
        for option in spec.specs:
            try:
                return convert(value, option, where)
            except ValueError:
                continue
        raise ValueError(f"{where}: expected {spec}, got {value!r}")

    if spec is NULL:
        if value != "null":
            raise ValueError(f"{where}: expected null, got {value!r}")
        return None

    if isinstance(spec, dict):
        return check(value, spec, where)

    if isinstance(spec, list):
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        return [
            convert(item, spec[0], f"{where}[{i}]") for i, item in enumerate(value)
        ]

    if not isinstance(value, str):
        raise ValueError(f"{where}: expected a single value, got {value!r}")

    if isinstance(spec, tuple):
        if value not in spec:
            raise ValueError(f"{where}: {value!r} is not one of {', '.join(spec)}")
        return value

    if spec is bool:
        if value not in ("true", "false"):
            raise ValueError(f"{where}: expected true or false, got {value!r}")
        return value == "true"

    try:
        return spec(value)
    except ValueError:
        raise ValueError(f"{where}: expected {spec.__name__}, got {value!r}") from None


def all_defaulted(spec: Any) -> bool:
    """Whether a spec is a group whose every setting has a default (or is such a group)."""
    if not isinstance(spec, dict) or not spec or "..." in spec or list(spec) == ["name"]:
        return False
    return all(
        (isinstance(field, tuple) and isinstance(field[0], type)) or all_defaulted(field)
        for field in spec.values()
    )


def check(values: Any, schema: dict, where: str) -> dict:
    """
    Check a mapping against its schema and convert every value.

    Parameters
    ----------
    values : Any
        Mapping as read from YAML.
    schema : dict
        Spec of every setting. A key ending in ``?`` is optional; a spec
        ``(type, value)`` is optional with that default, and a group whose
        settings all have defaults may be left out; a ``"..."`` key
        keeps any further keys unconverted; a schema whose only key is
        ``name``, mapping names to schemas, picks one by name.
    where : str
        Dotted location, for error messages; empty for the whole file.

    Returns
    -------
    dict
        Converted values, then any further keys as read.

    Raises
    ------
    ValueError
        On a non-mapping, an unknown name or setting, or a missing setting.
    """
    label = where or "run file"
    if not isinstance(values, dict):
        raise ValueError(f"{label}: expected a mapping, got {values!r}")

    if list(schema) == ["name"] and isinstance(schema["name"], dict):
        variants = schema["name"]
        name = values.get("name")
        if name not in variants:
            raise ValueError(f"{label}.name: unknown {name!r}; {suggestion(str(name), list(variants))}")
        return check(values, variants[name], where)

    open_ended = "..." in schema
    fields = {key.removesuffix("?"): spec for key, spec in schema.items() if key != "..."}
    optional = {key.removesuffix("?") for key in schema if key.endswith("?")}
    defaults = {
        key: spec[1]
        for key, spec in fields.items()
        if isinstance(spec, tuple) and isinstance(spec[0], type)
    }
    defaulted_groups = {key for key, spec in fields.items() if all_defaulted(spec)}
    valid = list(fields)
    for key in values:
        if key not in fields and not open_ended:
            raise ValueError(
                f"{label}: unknown setting {key!r}; {suggestion(key, valid)}"
            )
    for key in fields:
        if key not in values and key not in optional | set(defaults) | defaulted_groups:
            raise ValueError(f"{label}: missing setting {key!r}")

    prefix = f"{where}." if where else ""
    converted = {}
    for key, spec in fields.items():
        if key in defaults:
            converted[key] = (
                convert(values[key], spec[0], prefix + key)
                if key in values
                else defaults[key]
            )
        elif key in values:
            converted[key] = convert(values[key], spec, prefix + key)
        elif key in defaulted_groups:
            converted[key] = check({}, spec, prefix + key)
    converted.update({key: value for key, value in values.items() if key not in fields})
    return converted


def component_arguments(cls: Any, parameters: dict, provided: set, where: str) -> dict:
    """
    Check a component's parameters against its class or function and convert them.

    Parameters
    ----------
    cls : type or callable
        Component class (its ``__init__`` is read) or function.
    parameters : dict
        Parameters as read from YAML, without ``name``.
    provided : set of str
        Arguments the builder passes itself.
    where : str
        Dotted location, for error messages.

    Returns
    -------
    dict
        Parameters converted by the type hints.

    Raises
    ------
    ValueError
        On an unknown or missing parameter, or a value of the wrong type.
    """
    signature = inspect.signature(cls.__init__ if inspect.isclass(cls) else cls).parameters
    settable = {
        name: parameter
        for name, parameter in signature.items()
        if name != "self" and name not in provided
    }
    valid = list(settable)
    for key in parameters:
        if key not in settable:
            hint = suggestion(key, valid) if valid else "it takes no parameters"
            raise ValueError(f"{where}: unknown parameter {key!r}; {hint}")

    arguments = {}
    for name, parameter in settable.items():
        if name not in parameters:
            if parameter.default is inspect.Parameter.empty:
                if parameter.annotation is not inspect.Parameter.empty:
                    spec_from_hint(parameter.annotation, f"{where}.{name}")  # no YAML form: say so
                raise ValueError(f"{where}: missing parameter {name!r}")
            continue
        if parameter.annotation is inspect.Parameter.empty:
            raise ValueError(f"{where}: {cls.__name__}.{name} has no type hint")
        location = f"{where}.{name}"
        arguments[name] = convert(
            parameters[name], spec_from_hint(parameter.annotation, location), location
        )
    return arguments


def load_run(path: Path) -> dict:
    """
    Read, merge and check a run file.

    Parameters
    ----------
    path : Path
        Run file.

    Returns
    -------
    dict
        Every section, with values converted to their declared types.
    """
    return check(read_extending(Path(path), "runs"), SECTIONS, "")


def load_run_record(text: str) -> dict:
    """
    Check the resolved run file a checkpoint carries (complete, no ``extends``).

    Parameters
    ----------
    text : str
        ``resolved/run.yaml`` contents.

    Returns
    -------
    dict
        Every section, with values converted to their declared types.
    """
    return check(read_yaml_text(text, "checkpoint record run.yaml"), SECTIONS, "")
