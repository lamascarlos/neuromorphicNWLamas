"""Modelo ``stochastic1``: conmutación binaria ON/OFF con probabilidades por paso.

SET por sobretensión con ``P = P0·exp[alpha·(V - V_th)]`` y RESET con
``P = P_decay·exp(-|V|/V_th)``. Las probabilidades no escalan con
``TIME_STEP_DT``. Es el modelo por defecto.
"""

from typing import Any

import numpy as np

from .base import register_memristor_evol_model


@register_memristor_evol_model("stochastic1")
def stochastic_updater1(simulation: dict[str, Any], V_solved):
    """Actualiza estocásticamente el estado de los memristores.

    Vectorizado con NumPy sobre los arrays precalculados en
    ``simulation["circuit"]``.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto.

    Returns
    -------
    networkx.Graph
        El mismo objeto ``simulation["graph"]`` mutado in-place.

    Notes
    -----
    El orden de consumo del RNG difiere de la versión secuencial previa:
    con la misma semilla, los memristores que conmutan pueden no ser los
    mismos que antes. La distribución estadística del proceso es idéntica.

    See Also
    --------
    build_admittance_matrix : construye el sistema que produce ``V_solved``.
    """
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]

    mem_g = circuit["memristor_g"]
    mem_u = circuit["mem_u_idx"]
    mem_v = circuit["mem_v_idx"]
    mem_edges = circuit["mem_edge_keys"]

    V_th = p["V_THRESHOLD"]
    G_OFF = p["G_OFF"]
    G_ON = p["G_ON"]

    # Snapshot del estado ANTES de tomar decisiones: así una arista que
    # hace SET en este paso no puede hacer RESET en el mismo paso.
    is_off = mem_g == G_OFF
    is_on = mem_g == G_ON

    v_mem = np.abs(V_solved[mem_u] - V_solved[mem_v])

    # --- SET (OFF -> ON) ---
    set_candidates = is_off & (v_mem > V_th)
    n_set = int(set_candidates.sum())
    if n_set > 0:
        v_cand = v_mem[set_candidates]
        p_set = p["P0_SET"] * np.exp(p["ALPHA_SET"] * (v_cand - V_th))
        p_set = np.clip(p_set, 0.0, 1.0)
        r = np.random.random(n_set)
        cand_local = np.flatnonzero(r < p_set)
        cand_global = np.flatnonzero(set_candidates)[cand_local]

        mem_g[cand_global] = G_ON
        for j in cand_global:
            u, v = mem_edges[j]
            G.edges[u, v]["conductance"] = G_ON

    # --- RESET (ON -> OFF) ---
    reset_candidates = is_on
    n_reset = int(reset_candidates.sum())
    if n_reset > 0:
        v_cand = v_mem[reset_candidates]
        estabilidad = np.exp(-v_cand / V_th)
        p_decay = p["P_DECAY"] * estabilidad
        r = np.random.random(n_reset)
        cand_local = np.flatnonzero(r < p_decay)
        cand_global = np.flatnonzero(reset_candidates)[cand_local]

        mem_g[cand_global] = G_OFF
        for j in cand_global:
            u, v = mem_edges[j]
            G.edges[u, v]["conductance"] = G_OFF

    circuit["memristor_active"] = mem_g == G_ON
    return G
