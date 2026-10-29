"""Prior files: a preset, or a run-directory file that may extend one.

Presets live in ``sage/presets/priors/``. Prior ``min`` and ``max`` are
read as floats; every other value is a string.
"""

from pathlib import Path
from typing import Any, Dict

from sage.config.loader import read_extending, read_preset

TOP_LEVEL_KEYS = {"variable_params", "priors", "waveform_transforms", "constraints"}
REQUIRED_TOP_LEVEL_KEYS = {"variable_params", "priors"}
PRIOR_KEYS = {
    "uniform": {"min", "max"},
    "uniform_angle": set(),
    "sin_angle": set(),
    "uniform_sky": {"polar-angle", "azimuthal-angle"},
    "uniform_solidangle": {"polar-angle", "azimuthal-angle"},
    "uniform_radius": {"min", "max"},
    "uniform_comoving_volume": {"min", "max", "cosmology"},
}
TRANSFORM_KEYS = {
    "spherical_to_cartesian": {"x", "y", "z", "radial", "polar", "azimuthal"},
    "mass1_mass2_to_mchirp_q": set(),
    "chirp_distance_to_distance": set(),
}
CONSTRAINT_KEYS = {"params"}
# pycbc.distributions.sky_location.UniformSky: the angles' names when not given.
SKY_NAMES = {"polar-angle": "dec", "azimuthal-angle": "ra"}
NUMERIC_KEYS = {"min", "max"}


def check_keys(where: str, given, allowed: set) -> None:
    """
    Raise if ``given`` holds a key outside ``allowed``.

    Parameters
    ----------
    where : str
        Location named in the error message.
    given : iterable of str
        Keys found in the prior file.
    allowed : set of str
        Keys accepted at that location.

    Raises
    ------
    ValueError
        If any key is not accepted.
    """
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; allowed: {sorted(allowed)}")


def validate_prior(config: Dict[str, Any]) -> None:
    """
    Check a prior for unknown or inconsistent entries.

    Parameters
    ----------
    config : dict
        Parsed prior.

    Raises
    ------
    ValueError
        On an unknown key, distribution, transform or constraint, a missing
        required key, a transform used before its inputs exist, an output
        produced twice, or ``variable_params`` not matching the priors.
    """
    check_keys("the prior file", config, TOP_LEVEL_KEYS)
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(config))
    if missing:
        raise ValueError(f"prior file is missing {missing}")

    sampled = []
    for pname, pcfg in config["priors"].items():
        name = pcfg.get("name")
        if name not in PRIOR_KEYS:
            raise ValueError(
                f"unknown distribution {name!r} for prior {pname!r}; "
                f"allowed: {sorted(PRIOR_KEYS)}"
            )
        check_keys(f"prior {pname!r}", pcfg, PRIOR_KEYS[name] | {"name"})
        if name == "uniform_solidangle":
            sampled += [pcfg["polar-angle"], pcfg["azimuthal-angle"]]
        elif name == "uniform_sky":
            sampled += [pcfg.get(key, default) for key, default in SKY_NAMES.items()]
        else:
            sampled.append(pname)

    listed = set(config["variable_params"])
    if listed != set(sampled) or len(config["variable_params"]) != len(listed):
        raise ValueError(
            "variable_params does not match the parameters the priors draw: "
            f"missing {sorted(set(sampled) - listed)}, "
            f"not drawn {sorted(listed - set(sampled))}"
        )

    produced = list(sampled)
    for tname, tcfg in config.get("waveform_transforms", {}).items():
        name = tcfg.get("name")
        if name not in TRANSFORM_KEYS:
            raise ValueError(
                f"unknown transform {name!r} in {tname!r}; "
                f"allowed: {sorted(TRANSFORM_KEYS)}"
            )
        check_keys(f"transform {tname!r}", tcfg, TRANSFORM_KEYS[name] | {"name"})
        if name == "spherical_to_cartesian":
            needs = [tcfg["radial"], tcfg["polar"], tcfg["azimuthal"]]
            outputs = [tcfg["x"], tcfg["y"], tcfg["z"]]
        elif name == "mass1_mass2_to_mchirp_q":
            needs, outputs = ["mass1", "mass2"], ["mchirp", "q"]
        else:
            needs, outputs = ["chirp_distance", "mchirp"], ["distance"]
        absent = [n for n in needs if n not in produced]
        if absent:
            raise ValueError(
                f"transform {tname!r} needs {absent}, which no earlier prior or "
                "transform produces"
            )
        produced += outputs

    duplicates = sorted({n for n in produced if produced.count(n) > 1})
    if duplicates:
        raise ValueError(f"parameter(s) {duplicates} are produced more than once")

    for c in config.get("constraints", []):
        check_keys(f"constraint {c.get('name')!r}", c, CONSTRAINT_KEYS | {"name"})


def check_prior(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a prior and read its ``min`` and ``max`` as floats.

    Parameters
    ----------
    config : dict
        Prior as read from YAML, every scalar a string.

    Returns
    -------
    dict
        The same prior with numeric entries converted and the sky angles named.

    Raises
    ------
    ValueError
        If the prior is invalid or a ``min`` / ``max`` is not a number.
    """
    validate_prior(config)

    priors = {}
    for pname, pcfg in config["priors"].items():
        entry = {**SKY_NAMES, **pcfg} if pcfg["name"] == "uniform_sky" else dict(pcfg)
        for key in NUMERIC_KEYS & set(entry):
            try:
                entry[key] = float(entry[key])
            except ValueError:
                raise ValueError(
                    f"prior {pname!r}: {key} must be a number, got {entry[key]!r}"
                ) from None
        priors[pname] = entry
    return {**config, "priors": priors}


def load_prior_file(path: Path) -> Dict[str, Any]:
    """
    Read a prior file, merged onto the preset it extends, and check it.

    Parameters
    ----------
    path : Path
        Prior file.

    Returns
    -------
    dict
        Checked prior.
    """
    return check_prior(read_extending(Path(path), "priors"))


def load_prior(setting: str, run_dir: Path) -> Dict[str, Any]:
    """
    Load the prior a run's ``signal.prior`` setting names.

    Parameters
    ----------
    setting : str
        A preset name, or a ``.yaml`` file in the run directory.
    run_dir : Path
        Run directory.

    Returns
    -------
    dict
        Checked prior.
    """
    if setting.endswith(".yaml"):
        path = Path(run_dir) / setting
        if not path.is_file():
            raise FileNotFoundError(f"signal.prior: no prior file {path}")
        return load_prior_file(path)
    return check_prior(read_preset("priors", setting))
