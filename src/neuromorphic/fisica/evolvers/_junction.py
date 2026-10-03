"""Utilidades compartidas por los modelos de junturas Ag/PVP/Ag."""

from typing import Any

import numpy as np

G0 = 7.748091729e-5
"""Cuanto de conductancia 2e²/h (S)."""

KB = 8.617333262e-5
"""Constante de Boltzmann (eV/K)."""

JUNCTION_GEOMETRY_DEFAULTS: dict[str, Any] = {
    "GAP_D": 1.0e-9,  # m, espesor efectivo de PVP en la juntura
    "HOP_A": 2.5e-10,  # m, distancia de salto iónico
    "KAPPA": 5.0e9,  # 1/m, decaimiento túnel (phi ~ 1 eV)
    "N_MAX": 10,  # conductancia máxima = N_MAX * G0
}
"""Parámetros geométricos compartidos por ``ladder``, ``ladder_numba`` y ``thermal``."""


def make_ladder(
    d: float, a: float, kappa: float, n_max: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Escalera de conductancias de una juntura.

    Returns
    -------
    ladder : numpy.ndarray
        Niveles: túnel ``G0·exp(-2κ·gap)`` con el gap reducido de a un salto
        ``a``, seguidos de los niveles metálicos ``n·G0``, ``n = 1..n_max``.
    gfac : numpy.ndarray
        ``gap / d`` de cada nivel (en los metálicos se usa ``a / d``).
    edges : numpy.ndarray
        Fronteras entre niveles, en escala logarítmica, para recuperar el
        nivel ``k`` a partir de ``G`` con ``searchsorted``.
    """
    n_t = round(d / a)
    gaps = d - a * np.arange(n_t)
    ladder = np.concatenate([G0 * np.exp(-2.0 * kappa * gaps), G0 * np.arange(1, n_max + 1)])
    gfac = np.concatenate([gaps, np.full(n_max, a)]) / d
    edges = np.sqrt(ladder[:-1] * ladder[1:])
    return ladder, gfac, edges


def filament_closed(mem_g: np.ndarray) -> np.ndarray:
    """Criterio de juntura activa: el filamento cierra el gap (``G >= G0``)."""
    return mem_g >= G0 * (1.0 - 1e-9)


def require_scalar(p: dict[str, Any], keys) -> None:
    """Verifica que los parámetros indicados sean escalares."""
    for key in keys:
        if np.ndim(p[key]) != 0:
            raise ValueError(
                f"El parámetro {key} debe ser escalar en este modelo "
                f"(recibido con shape {np.shape(p[key])})."
            )


def set_conductances(simulation: dict[str, Any], new_g: np.ndarray) -> None:
    """Escribe ``new_g`` en ``memristor_g``, el espejo del grafo y la máscara.

    Solo recorre en Python las aristas cuyo valor cambió.
    """
    circuit = simulation["circuit"]
    mem_g = circuit["memristor_g"]
    changed = np.flatnonzero(new_g != mem_g)
    mem_g[:] = new_g
    G = simulation["graph"]
    keys = circuit["mem_edge_keys"]
    for j in changed:
        u, v = keys[j]
        G.edges[u, v]["conductance"] = float(mem_g[j])
    circuit["memristor_active"] = filament_closed(mem_g)


def junction_drive(simulation: dict[str, Any], V_solved: np.ndarray):
    """Caída de tensión ``|V|`` y corriente ``G·V`` sobre cada juntura."""
    circuit = simulation["circuit"]
    v_mem = np.abs(V_solved[circuit["mem_u_idx"]] - V_solved[circuit["mem_v_idx"]])
    i_mem = circuit["memristor_g"] * v_mem
    return v_mem, i_mem
