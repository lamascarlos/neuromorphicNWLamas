"""Modelo ``ladder_numba``: el proceso de ``ladder`` con un kernel Numba.

Requiere el extra opcional ``numba`` (``pip install "neuromorphic-nwlamas[numba]"``).
El modelo se registra siempre; si numba no está instalado, falla al
inicializarse con un ``ImportError`` explicativo.

Diferencia con ``ladder``: el truncamiento en ``LADDER_MAX_JUMPS`` se
aplica al número total de eventos (y luego se sortea la dirección de cada
uno), mientras que ``ladder`` trunca subidas y bajadas por separado. Ambos
coinciden en distribución cuando ``r·dt`` es chico. La ruptura tiene
prioridad sobre los saltos en el mismo paso.
"""

from typing import Any

import numpy as np

from ._junction import junction_drive, set_conductances
from .base import register_memristor_evol_model
from .ladder import LADDER_PARAMETERS, LADDER_RATE_DEFAULTS, build_ladder_state

_RATE_KEYS = list(LADDER_RATE_DEFAULTS)


def _kernel():
    try:
        from . import _ladder_numba_kernel
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "El modelo 'ladder_numba' requiere numba. Instalalo con "
            '`pip install "neuromorphic-nwlamas[numba]"` o usá EVOLVER="ladder".'
        ) from exc
    return _ladder_numba_kernel


def init_ladder_numba(simulation: dict[str, Any]) -> None:
    kernel = _kernel()
    build_ladder_state(simulation)
    p = simulation["parameters"]
    n = simulation["circuit"]["memristor_g"].size
    rates = np.empty((len(_RATE_KEYS), n), dtype=np.float64)
    for row, key in enumerate(_RATE_KEYS):
        rates[row] = np.broadcast_to(np.asarray(p[key], dtype=np.float64), (n,))
    state = simulation["evolver_state"]
    state["rates"] = rates
    state["buffer"] = np.empty(n, dtype=np.float64)
    kernel.seed(int(p["RNG_SEED"]))


@register_memristor_evol_model("ladder_numba", init=init_ladder_numba, parameters=LADDER_PARAMETERS)
def ladder_numba_updater(simulation: dict[str, Any], V_solved):
    """Un paso del proceso de nacimiento-muerte (kernel Numba).

    Los parámetros de tasa se leen al inicializar: cambiarlos en
    ``simulation["parameters"]`` a mitad de corrida no tiene efecto.
    """
    kernel = _kernel()
    p = simulation["parameters"]
    state = simulation["evolver_state"]
    mem_g = simulation["circuit"]["memristor_g"]
    v_mem, i_mem = junction_drive(simulation, V_solved)
    out = state["buffer"]
    kernel.ladder_step(
        mem_g,
        v_mem,
        i_mem,
        float(p["TIME_STEP_DT"]),
        state["ladder"],
        state["gfac"],
        state["edges"],
        state["rates"],
        out,
    )
    set_conductances(simulation, out)
    return simulation["graph"]
