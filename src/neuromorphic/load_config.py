# load_config.py
"""Configuration loading for the neuromorphic nanowire simulator.

Design
------
- ``_load_defaults()`` reads the ``.ini`` bundled with the package.
- ``_load_ini(path)`` reads an explicit user-provided ``.ini``.
- ``load_parameters(filepath=None, parms=None)`` is the single public
  entry point. It never searches the current working directory
  implicitly: if ``filepath`` is None, only the package defaults are used.

Keys in ``parms`` are always merged into the parameter dict. Keys that are
neither core keys (:data:`KNOWN_RAW_KEYS`) nor declared by a registered
evolution model trigger an ``UnknownParameterWarning`` (to surface typos),
but are kept: models and user code may read extra parameters.

Besides the fixed sections, an ``.ini`` may contain an optional
``[Evolver]`` section with free-form keys. Each key is upper-cased and its
value parsed as a Python literal when possible (numbers, lists, booleans),
falling back to the raw string.
"""

import ast
import configparser
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INI_NAME = "defaults.ini"


class UnknownParameterWarning(UserWarning):
    """Emitted when ``parms`` contains keys nobody declared (likely a typo)."""


# Core keys of the parameter dict. Kept explicit so that typos in ``parms``
# are surfaced as warnings. Evolution models declare their own keys at
# registration (see ``register_memristor_evol_model``).
KNOWN_RAW_KEYS: frozenset[str] = frozenset(
    {
        # Network
        "NUM_WIRES",
        "AREA",
        "LENGTH",
        "PROXIMITY_THRESHOLD_RATIO",
        # Electrical
        "V_INPUT",
        "V_GROUND",
        "V_READ",
        "RHO_PLATA",
        "DIAMETRO_NM",
        # Time
        "TIME_STEP_DT",
        "T_PULSE",
        "T_RELAX",
        # Memristor
        "G_ON",
        "G_OFF",
        "V_THRESHOLD",
        "EVOLVER",
        # Probabilities
        "P0_SET",
        "ALPHA_SET",
        "P_DECAY",
        "BETA_RESET",
        "RNG_SEED",
    }
)

DERIVED_KEYS: frozenset[str] = frozenset(
    {
        "PROXIMITY_THRESHOLD",
        "R_WIRE_PER_LENGTH",
        "TOTAL_TIME",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_parameters(
    filepath: str | Path | None = None,
    parms: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Carga los parámetros de simulación.

    Parameters
    ----------
    filepath : str or pathlib.Path, optional
        Ruta a un archivo ``.ini`` explícito. Si es ``None`` (default),
        se usa el ``defaults.ini`` empaquetado con la librería. No hay
        búsqueda implícita en el directorio actual.
    parms : dict, optional
        Overrides y parámetros adicionales, aplicados sobre los valores
        del archivo. Todas las claves se incorporan. Las que no son
        claves del núcleo ni fueron declaradas por un modelo de evolución
        registrado disparan :class:`UnknownParameterWarning`. Los valores
        derivados se recalculan automáticamente después del merge.

    Returns
    -------
    dict
        Diccionario con la clave ``"parameters"`` que contiene tanto los
        valores crudos del ``.ini`` como los derivados:
        ``PROXIMITY_THRESHOLD``, ``R_WIRE_PER_LENGTH`` y ``TOTAL_TIME``.

    Raises
    ------
    FileNotFoundError
        Si ``filepath`` es explícito y no existe, o si el ``defaults.ini``
        empaquetado no se encuentra.

    Warns
    -----
    UnknownParameterWarning
        Si ``parms`` contiene claves que no pertenecen a
        :data:`KNOWN_RAW_KEYS` ni a los parámetros declarados por los
        modelos de evolución registrados.

    See Also
    --------
    setup_simulation : construye el dict completo de simulación.
    """
    params = _load_defaults() if filepath is None else _load_ini(Path(filepath))

    if parms is not None:
        from .fisica.evolvers import declared_evolver_parameters

        unknown = set(parms) - KNOWN_RAW_KEYS - declared_evolver_parameters()
        if unknown:
            warnings.warn(
                f"Undeclared parameter(s) {sorted(unknown)} were added. "
                "Check for typos; known core keys: "
                f"{sorted(KNOWN_RAW_KEYS)}",
                UnknownParameterWarning,
                stacklevel=2,
            )
        params.update(parms)

    _recompute_derived(params)
    return {"parameters": params}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_defaults() -> dict[str, Any]:
    """Read the packaged ``defaults.ini`` via importlib.resources."""
    ini = files("neuromorphic").joinpath(DEFAULT_INI_NAME)
    if not ini.is_file():
        raise FileNotFoundError(
            f"Packaged {DEFAULT_INI_NAME} not found. The package is likely broken; reinstall it."
        )
    with ini.open("r", encoding="utf-8") as fh:
        return _parse_ini(fh)


def _load_ini(path: Path) -> dict[str, Any]:
    """Read a user-provided ``.ini`` from an explicit path."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return _parse_ini(fh)


def _parse_ini(fh) -> dict[str, Any]:
    config = configparser.ConfigParser()
    config.read_file(fh)

    params: dict[str, Any] = {}

    # Network
    params["NUM_WIRES"] = config.getint("Network", "num_wires")
    params["AREA"] = config.getfloat("Network", "area")
    params["LENGTH"] = config.getfloat("Network", "length")
    params["PROXIMITY_THRESHOLD_RATIO"] = config.getfloat("Network", "proximity_threshold_ratio")

    # Electrical
    params["G_LEAK"] = config.getfloat("Electrical", "g_leak")
    params["V_INPUT"] = config.getfloat("Electrical", "v_input")
    params["V_GROUND"] = config.getfloat("Electrical", "v_ground")
    params["V_READ"] = config.getfloat("Electrical", "v_read")
    params["RHO_PLATA"] = config.getfloat("Electrical", "rho_plata")
    params["DIAMETRO_NM"] = config.getfloat("Electrical", "diametro_nm")

    # Time
    params["TIME_STEP_DT"] = config.getfloat("Time", "time_step_dt")
    params["T_PULSE"] = config.getfloat("Time", "t_pulse")
    params["T_RELAX"] = config.getfloat("Time", "t_relax")

    # Memristor
    params["G_ON"] = config.getfloat("Memristor", "g_on")
    params["G_OFF"] = config.getfloat("Memristor", "g_off")
    params["V_THRESHOLD"] = config.getfloat("Memristor", "v_threshold")
    params["EVOLVER"] = config.get("Memristor", "evolver_model")

    # Probabilities
    params["P0_SET"] = config.getfloat("Probabilities", "p0_set")
    params["ALPHA_SET"] = config.getfloat("Probabilities", "alpha_set")
    params["P_DECAY"] = config.getfloat("Probabilities", "p_decay")
    params["BETA_RESET"] = config.getfloat("Probabilities", "beta_reset")
    params["RNG_SEED"] = config.getint("Probabilities", "seed")

    # Optional free-form section for evolution-model parameters
    if config.has_section("Evolver"):
        for key, raw in config.items("Evolver"):
            params[key.upper()] = _parse_value(raw)

    return params


def _parse_value(raw: str) -> Any:
    """Parse an ``.ini`` value as a Python literal, else return the string."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _recompute_derived(params: dict[str, Any]) -> None:
    """Recalculate derived values in-place from raw keys."""
    params["PROXIMITY_THRESHOLD"] = params["PROXIMITY_THRESHOLD_RATIO"] * params["AREA"]
    radio_um = (params["DIAMETRO_NM"] / 2.0) * 1e-3
    params["R_WIRE_PER_LENGTH"] = params["RHO_PLATA"] / (np.pi * radio_um**2)
    params["TOTAL_TIME"] = params["T_PULSE"] + params["T_RELAX"]
