"""Modelo ``stochastic2``: conmutación binaria ON/OFF de Lamas et al. (2026).

SET con ``P = Δt·P0·max(0, 1 - exp[-alpha·(V - V_th)])`` y RESET con
``P = min(1, Δt·P_decay·I²)``, ``I = G_ON·V``.
"""

from typing import Any

import numpy as np

from .base import register_memristor_evol_model


@register_memristor_evol_model("stochastic2")
def stochastic_updater2(simulation: dict[str, Any], V_solved):
    r"""Actualiza los memristores según el modelo de Lamas et al. (2026).

    Implementa las probabilidades de conmutación por paso temporal del
    paper:

    .. math::

        P_\mathrm{set} = \Delta t \, P_0 \,
            \max\!\left(0,\; 1 - e^{-\alpha (V_\mathrm{mem} - V_\mathrm{th})}\right)

        P_\mathrm{reset} = \min\!\left(1,\; \Delta t \, P_\mathrm{decay} \, I_\mathrm{mem}^2\right)

    donde :math:`V_\mathrm{mem} = |V_u - V_v|` es la caída de tensión
    sobre la juntura e :math:`I_\mathrm{mem} = G_\mathrm{mem} V_\mathrm{mem}`
    la corriente que la atraviesa.

    A diferencia de ``stochastic1``, las probabilidades escalan con
    ``TIME_STEP_DT``, de modo que el resultado estadístico no depende del
    paso temporal (para :math:`\Delta t` chico). ``P0_SET`` tiene unidades
    de s⁻¹ y ``P_DECAY`` de A⁻² s⁻¹.

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
    - Las decisiones se toman sobre el estado al inicio del paso: una
      juntura que hace SET no puede hacer RESET en el mismo paso.
    - :math:`P_\mathrm{set}` se recorta a 1 por seguridad; con
      :math:`\Delta t \, P_0 > 1` el modelo deja de ser una probabilidad
      por paso válida y conviene reducir ``TIME_STEP_DT``.
    - ``BETA_RESET`` no interviene en este modelo.

    See Also
    --------
    neuromorphic.fisica.evolvers.stochastic1.stochastic_updater1
        Modelo previo, sin dependencia en Δt ni en la corriente.
    """
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]

    mem_g = circuit["memristor_g"]
    mem_edges = circuit["mem_edge_keys"]

    dt = p["TIME_STEP_DT"]
    V_th = p["V_THRESHOLD"]
    G_OFF = p["G_OFF"]
    G_ON = p["G_ON"]

    # Snapshot del estado al inicio del paso.
    is_off = mem_g == G_OFF
    is_on = mem_g == G_ON

    v_mem = np.abs(V_solved[circuit["mem_u_idx"]] - V_solved[circuit["mem_v_idx"]])

    # --- SET (OFF -> ON): activado por voltaje ---
    set_candidates = np.flatnonzero(is_off & (v_mem > V_th))
    if set_candidates.size > 0:
        overdrive = v_mem[set_candidates] - V_th
        p_set = dt * p["P0_SET"] * (-np.expm1(-p["ALPHA_SET"] * overdrive))
        p_set = np.clip(p_set, 0.0, 1.0)
        switched = set_candidates[np.random.random(set_candidates.size) < p_set]
        _apply_switch(mem_g, G, mem_edges, switched, G_ON)

    # --- RESET (ON -> OFF): activado por corriente (disipación Joule) ---
    reset_candidates = np.flatnonzero(is_on)
    if reset_candidates.size > 0:
        i_mem = G_ON * v_mem[reset_candidates]
        p_reset = np.minimum(1.0, dt * p["P_DECAY"] * i_mem**2)
        switched = reset_candidates[np.random.random(reset_candidates.size) < p_reset]
        _apply_switch(mem_g, G, mem_edges, switched, G_OFF)

    circuit["memristor_active"] = mem_g == G_ON
    return G


def _apply_switch(mem_g, G, mem_edges, indices, value: float) -> None:
    """Fija ``value`` en ``mem_g[indices]`` y en el espejo del grafo."""
    mem_g[indices] = value
    for j in indices:
        u, v = mem_edges[j]
        G.edges[u, v]["conductance"] = value
