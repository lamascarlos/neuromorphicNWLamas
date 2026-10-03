"""Infraestructura del registro de modelos de evolución de memristores.

Un modelo de evolución consta de:

- ``update(simulation, V_solved)``: avanza un paso temporal. Debe mantener
  coherentes ``circuit["memristor_g"]``, el espejo
  ``graph.edges[u, v]["conductance"]`` y la máscara
  ``circuit["memristor_active"]``.
- ``init(simulation)`` (opcional): fija el estado inicial de los
  memristores y guarda lo que el modelo necesite precalcular en
  ``simulation["evolver_state"]``. Por defecto, todos en ``G_OFF``.
- ``parameters`` (opcional): parámetros propios del modelo con sus valores
  por defecto. Se aceptan en ``parms=`` sin advertencias y se completan en
  ``simulation["parameters"]`` al inicializar el modelo.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

EvolverFn = Callable[[dict[str, Any], np.ndarray], Any]
InitFn = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class EvolverSpec:
    """Especificación completa de un modelo de evolución registrado."""

    update: EvolverFn
    init: InitFn
    parameters: Mapping[str, Any] = field(default_factory=dict)


EVOLVER_SPECS: dict[str, EvolverSpec] = {}
"""Nombre -> :class:`EvolverSpec` completa."""

_F = TypeVar("_F", bound=EvolverFn)


def set_all_memristors(simulation: dict[str, Any], g_value: float) -> None:
    """Fija todos los memristores en ``g_value`` (array, grafo y máscara)."""
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]
    circuit["memristor_g"][:] = g_value
    for u, v in circuit["mem_edge_keys"]:
        G.edges[u, v]["conductance"] = g_value
    circuit["memristor_active"] = circuit["memristor_g"] == p["G_ON"]


def init_all_off(simulation: dict[str, Any]) -> None:
    """Inicialización por defecto: todos los memristores en ``G_OFF``."""
    set_all_memristors(simulation, simulation["parameters"]["G_OFF"])


def register_memristor_evol_model(
    name: str,
    *,
    init: InitFn | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Callable[[_F], _F]:
    """Decorador que registra un modelo de evolución bajo ``name``.

    Parameters
    ----------
    name : str
        Nombre con el que se selecciona el modelo (``EVOLVER``).
    init : callable, optional
        ``init(simulation)``. Si se omite, se usa :func:`init_all_off`.
    parameters : mapping, optional
        Parámetros propios del modelo y sus valores por defecto.

    Registrar un nombre ya existente lo sobrescribe sin aviso.
    """

    def _register(fn: _F) -> _F:
        EVOLVER_SPECS[name] = EvolverSpec(
            update=fn,
            init=init if init is not None else init_all_off,
            parameters=dict(parameters or {}),
        )
        return fn

    return _register


def declared_evolver_parameters() -> frozenset[str]:
    """Claves de parámetros declaradas por algún modelo registrado."""
    return frozenset(k for spec in EVOLVER_SPECS.values() for k in spec.parameters)


def initialize_evolver(simulation: dict[str, Any]) -> EvolverSpec:
    """Prepara el modelo seleccionado en ``parameters["EVOLVER"]``.

    Completa en ``simulation["parameters"]`` los parámetros del modelo que
    no estén definidos, recrea ``simulation["evolver_state"]`` vacío y
    llama a ``init``. Es idempotente: llamarla de nuevo reinicia el estado
    de los memristores.

    Raises
    ------
    KeyError
        Si ``EVOLVER`` no corresponde a ningún modelo registrado.
    """
    p = simulation["parameters"]
    name = p["EVOLVER"]
    if name not in EVOLVER_SPECS:
        raise KeyError(
            f"Modelo de evolución desconocido: {name!r}. Registrados: {sorted(EVOLVER_SPECS)}"
        )
    spec = EVOLVER_SPECS[name]
    for key, default in spec.parameters.items():
        p.setdefault(key, default)
    simulation["evolver_state"] = {}
    spec.init(simulation)
    return spec
