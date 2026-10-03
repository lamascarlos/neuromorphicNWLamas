# simulation.py
"""High-level entry point that assembles a simulation dict."""

from typing import Any

import numpy as np

from .fisica import build_admittance_matrix, initialize_evolver
from .geometria import generate_and_find_junctions
from .grafo import build_graph, find_electrode_nodes, prune_dead_components
from .load_config import load_parameters


def setup_simulation(
    filepath: str | None = None,
    parms: dict[str, Any] | None = None,
    prune: bool = True,
) -> dict[str, Any]:
    """Construye el diccionario completo de simulación.

    Encadena la carga de parámetros, la generación geométrica de la red,
    la construcción del grafo, la detección de electrodos y el ensamblado
    inicial de la matriz de admitancia.

    Parameters
    ----------
    filepath : str, optional
        Ruta a un ``.ini`` explícito. Si es ``None``, se usa el
        ``defaults.ini`` empaquetado.
    parms : dict, optional
        Overrides sobre los parámetros cargados. Las claves desconocidas
        disparan :class:`~neuromorphic.load_config.UnknownParameterWarning`
        y se ignoran.

    prune: bool
        Si es True, poda los sub-grafos desconectados del resto.
        Default: ``True``.

    Returns
    -------
    dict
        Diccionario con las claves:

        - ``"parameters"`` : parámetros crudos y derivados.
        - ``"junctions"`` : estructuras geométricas de la red.
        - ``"graph"`` : objeto :class:`networkx.Graph`.
        - ``"terminals"`` : ``{"input_nodes", "output_nodes"}``.
        - ``"circuit"`` : matriz ``Y``, vector ``I``, ``node_to_index`` y
          los arrays por memristor (``memristor_g``, ``memristor_active``,
          ``mem_edge_keys``, ...).
        - ``"evolver_state"`` : estado propio del modelo de evolución.

    See Also
    --------
    load_parameters : carga y valida los parámetros.
    run_simulation_dynamic_pulse : corre el experimento de pulso.
    """
    simulation = load_parameters(filepath, parms=parms)
    np.random.seed(simulation["parameters"]["RNG_SEED"])
    generate_and_find_junctions(simulation)
    build_graph(simulation)
    find_electrode_nodes(simulation)
    if prune:
        prune_dead_components(simulation)
    build_admittance_matrix(simulation)
    initialize_evolver(simulation)
    return simulation
