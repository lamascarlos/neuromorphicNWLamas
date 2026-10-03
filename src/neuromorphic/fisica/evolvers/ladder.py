"""Modelo ``ladder``: proceso de nacimiento-muerte sobre una escalera de G.

Reformulación de ``stochastic_junction.py``. El estado de cada juntura es
su conductancia, restringida a una escalera discreta de niveles (ver
:func:`~neuromorphic.fisica.evolvers._junction.make_ladder`). En cada paso ``dt``:

- saltos hacia arriba ~ ``Poisson(r_up·dt)``, ``r_up = nu_up·sinh(|V| / V_k)``,
  con ``V_k = V_s·gap_k/d`` (el campo en el gap crece al cerrarse);
- saltos hacia abajo ~ ``Poisson(r_dn·dt)``, ``r_dn = nu_dn·exp(-beta·k)·exp(P/P_T)``;
- ruptura (``k -> 0``) con tasa ``r_rp = nu_rp·(P/P_c)^m``, ``P = |I·V|``.

El número de saltos en cada dirección se trunca en ``LADDER_MAX_JUMPS``.
El nivel ``k`` se recupera de ``G``; no hay variables internas por juntura.
"""

from typing import Any

import numpy as np

from ._junction import (
    JUNCTION_GEOMETRY_DEFAULTS,
    filament_closed,
    junction_drive,
    make_ladder,
    require_scalar,
    set_conductances,
)
from .base import register_memristor_evol_model, set_all_memristors

LADDER_RATE_DEFAULTS: dict[str, Any] = {
    "LADDER_NU_UP": 1e3,  # 1/s, crecimiento
    "LADDER_V_S": 0.2,  # V
    "LADDER_NU_DN": 1.0,  # 1/s, disolución
    "LADDER_BETA": 0.7,
    "LADDER_P_T": 1e-5,  # W
    "LADDER_NU_RP": 1e3,  # 1/s, ruptura
    "LADDER_P_C": 6e-5,  # W
    "LADDER_M": 6,
    "LADDER_MAX_JUMPS": 2,
}

LADDER_PARAMETERS: dict[str, Any] = {**JUNCTION_GEOMETRY_DEFAULTS, **LADDER_RATE_DEFAULTS}
"""Parámetros de ``ladder`` y ``ladder_numba``. Las tasas pueden ser arrays por juntura."""


def build_ladder_state(simulation: dict[str, Any]) -> None:
    """Precalcula la escalera y deja todas las junturas en el nivel más bajo."""
    p = simulation["parameters"]
    require_scalar(p, JUNCTION_GEOMETRY_DEFAULTS)
    require_scalar(p, ["LADDER_MAX_JUMPS"])
    ladder, gfac, edges = make_ladder(p["GAP_D"], p["HOP_A"], p["KAPPA"], int(p["N_MAX"]))
    state = simulation["evolver_state"]
    state["ladder"] = ladder
    state["gfac"] = gfac
    state["edges"] = edges
    set_all_memristors(simulation, ladder[0])
    simulation["circuit"]["memristor_active"] = filament_closed(
        simulation["circuit"]["memristor_g"]
    )


def init_ladder(simulation: dict[str, Any]) -> None:
    build_ladder_state(simulation)
    p = simulation["parameters"]
    simulation["evolver_state"]["rng"] = np.random.default_rng(p["RNG_SEED"])


@register_memristor_evol_model("ladder", init=init_ladder, parameters=LADDER_PARAMETERS)
def ladder_updater(simulation: dict[str, Any], V_solved):
    """Un paso del proceso de nacimiento-muerte (versión NumPy).

    Los parámetros de tasa (``LADDER_NU_UP``, ``LADDER_V_S``, ...) pueden
    ser escalares o arrays con un valor por juntura, en el orden de
    ``memristor_g``.
    """
    p = simulation["parameters"]
    state = simulation["evolver_state"]
    ladder, gfac, edges, rng = state["ladder"], state["gfac"], state["edges"], state["rng"]
    dt = p["TIME_STEP_DT"]
    kmax = len(ladder) - 1
    max_jumps = int(p["LADDER_MAX_JUMPS"])

    mem_g = simulation["circuit"]["memristor_g"]
    k = np.searchsorted(edges, mem_g)
    v_mem, i_mem = junction_drive(simulation, V_solved)
    power = np.abs(i_mem * v_mem)

    r_up = p["LADDER_NU_UP"] * np.sinh(np.minimum(v_mem / (p["LADDER_V_S"] * gfac[k]), 50.0))
    r_dn = (
        p["LADDER_NU_DN"]
        * np.exp(-p["LADDER_BETA"] * k)
        * np.exp(np.minimum(power / p["LADDER_P_T"], 50.0))
    )
    r_rp = p["LADDER_NU_RP"] * (power / p["LADDER_P_C"]) ** p["LADDER_M"]

    n_up = np.minimum(rng.poisson(np.minimum(r_up * dt, 1e3)), max_jumps)
    n_dn = np.minimum(rng.poisson(np.minimum(r_dn * dt, 1e3)), max_jumps)
    k_new = np.clip(k + n_up - n_dn, 0, kmax)

    rupt = rng.random(k.shape) < -np.expm1(-r_rp * dt)
    k_new = np.where(rupt, 0, k_new)

    set_conductances(simulation, ladder[k_new])
    return simulation["graph"]
