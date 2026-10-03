"""Registro de modelos de evolución de memristores.

Al importar este paquete se importan automáticamente todos sus módulos
públicos (los que no empiezan con ``_``), y cada uno registra sus modelos
con :func:`register_memristor_evol_model`. Para agregar un modelo al
paquete basta con crear un módulo nuevo en este directorio.

Los módulos privados (``_junction``, ``_ladder_numba_kernel``, ...) no se
importan automáticamente: sirven para utilidades compartidas y para
código con dependencias opcionales que solo se carga al usarse.
"""

import importlib
import pkgutil

from .base import (
    EVOLVER_SPECS,
    EvolverSpec,
    declared_evolver_parameters,
    init_all_off,
    initialize_evolver,
    register_memristor_evol_model,
    set_all_memristors,
)


def _discover_models() -> list[str]:
    """Importa los módulos públicos del paquete y devuelve sus nombres."""
    names = sorted(
        info.name
        for info in pkgutil.iter_modules(__path__)
        if not info.name.startswith("_") and info.name != "base"
    )
    for name in names:
        importlib.import_module(f"{__name__}.{name}")
    return names


MODEL_MODULES: list[str] = _discover_models()
"""Módulos de modelos importados automáticamente, en orden alfabético."""

__all__ = [
    "EVOLVER_SPECS",
    "MODEL_MODULES",
    "EvolverSpec",
    "declared_evolver_parameters",
    "init_all_off",
    "initialize_evolver",
    "register_memristor_evol_model",
    "set_all_memristors",
]
