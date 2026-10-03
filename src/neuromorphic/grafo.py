# grafo.py

from typing import Any

import networkx as nx
import numpy as np


# ==============================================================================
# CONSTRUCCIÓN DEL GRAFO
# ==============================================================================
def build_graph(simulation: dict[str, Any]) -> None:
    """Construye el grafo topológico de la red con duplicación de nodos.

    Por cada juntura física se crean dos nodos en el grafo (uno por cada
    cara del nanohilo que cruza). La arista que los une representa el
    memristor y se inicializa en estado ``G_OFF``. Los segmentos internos
    de cada nanohilo se modelan como aristas resistivas con ``weight``
    igual a la distancia entre junturas consecutivas.

    Parameters
    ----------
    simulation : dict[str, Any]
        Diccionario de simulación. Debe contener ``"parameters"`` con la
        clave ``G_OFF`` y ``"junctions"`` con el sub-diccionario generado
        por :func:`~neuromorphic.geometria.generate_and_find_junctions`.

    Returns
    -------
    None
        Modifica ``simulation`` in-place agregando la clave ``"graph"`` con
        un objeto :class:`networkx.Graph`. Cada nodo lleva el atributo
        ``pos`` (coordenadas 2D). Las aristas memristivas tienen
        ``is_memristor=True`` y ``conductance``; las resistivas tienen
        ``weight``.
    """
    p = simulation["parameters"]
    junction_data = simulation["junctions"]
    wires = junction_data["wires"]
    junctions = junction_data["junctions"]
    wire_to_junctions = junction_data["wire_to_junctions"]

    graph: nx.Graph = nx.Graph()
    wire_junction_to_graph_node = {}

    # Crear dos nodos por unión y la arista memristiva entre ellos
    for j_data in junctions:
        orig_junction_id = j_data["id"]
        w1_id, w2_id = j_data["wires"]

        node_for_w1_side = 2 * orig_junction_id + 1
        node_for_w2_side = 2 * orig_junction_id

        wire_junction_to_graph_node[(orig_junction_id, w1_id)] = node_for_w1_side
        wire_junction_to_graph_node[(orig_junction_id, w2_id)] = node_for_w2_side

        graph.add_node(node_for_w1_side, pos=j_data["pos"])
        graph.add_node(node_for_w2_side, pos=j_data["pos"])

        graph.add_edge(
            node_for_w1_side, node_for_w2_side, is_memristor=True, conductance=p["G_OFF"]
        )

    # Segmentos internos de cada nanohilo (resistencias lineales)
    for wire_id, junctions_list_for_wire in wire_to_junctions.items():
        if len(junctions_list_for_wire) < 2:
            continue

        start_point = wires[wire_id]["p1"]
        junctions_with_dist = []
        for j in junctions_list_for_wire:
            dist = np.linalg.norm(j["pos"] - start_point)
            junctions_with_dist.append((dist, j))
        junctions_with_dist.sort(key=lambda x: x[0])

        for k in range(len(junctions_with_dist) - 1):
            u_orig_junction_data = junctions_with_dist[k][1]
            v_orig_junction_data = junctions_with_dist[k + 1][1]
            dist_between = junctions_with_dist[k + 1][0] - junctions_with_dist[k][0]

            u_graph_node = wire_junction_to_graph_node[(u_orig_junction_data["id"], wire_id)]
            v_graph_node = wire_junction_to_graph_node[(v_orig_junction_data["id"], wire_id)]

            graph.add_edge(u_graph_node, v_graph_node, weight=dist_between)

    simulation["graph"] = graph
    return


# ==============================================================================
# DETECCIÓN DE ELECTRODOS
# ==============================================================================
def find_electrode_nodes(simulation: dict[str, Any]) -> None:
    """Identifica los nodos del grafo que actúan como electrodos.

    Un nodo es electrodo de entrada si su coordenada ``x`` es menor que
    ``PROXIMITY_THRESHOLD``; es electrodo de salida si ``x`` es mayor que
    ``AREA - PROXIMITY_THRESHOLD``.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"graph"`` y
        ``"parameters"`` con las claves ``AREA`` y ``PROXIMITY_THRESHOLD``.

    Returns
    -------
    None
        Modifica ``simulation`` in-place agregando la clave ``"terminals"``
        con ``{"input_nodes": list, "output_nodes": list}``.
    """
    G = simulation["graph"]
    p = simulation["parameters"]
    area_size = p["AREA"]
    threshold = p["PROXIMITY_THRESHOLD"]

    pos = nx.get_node_attributes(G, "pos")
    input_nodes, output_nodes = [], []
    for node_id, coords in pos.items():
        x = coords[0]
        if x < threshold:
            input_nodes.append(node_id)
        elif x > area_size - threshold:
            output_nodes.append(node_id)

    simulation["terminals"] = {
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
    }
    return


# ==============================================================================
# ELIMINA SUB-GRAFOS DESCONECTADOS
# ==============================================================================
def prune_dead_components(simulation: dict) -> int:
    """Elimina del grafo los nodos que no están conectados a ningún electrodo.

    Un nodo es "vivo" si pertenece a la misma componente conexa que al
    menos un nodo de entrada o de salida. Los nodos flotantes no
    transportan corriente y vuelven la matriz de admitancia singular
    cuando ``G_LEAK`` es muy pequeño.

    Parameters
    ----------
    simulation : dict
        Diccionario con ``"graph"`` y ``"terminals"``.

    Returns
    -------
    int
        Cantidad de nodos eliminados.
    """
    G = simulation["graph"]
    terminals = simulation["terminals"]

    live_seeds = set(terminals["input_nodes"]) | set(terminals["output_nodes"])
    if not live_seeds:
        return 0

    keep = set()
    for seed in live_seeds:
        keep |= nx.node_connected_component(G, seed)

    to_remove = set(G.nodes) - keep
    if to_remove:
        G.remove_nodes_from(to_remove)

        input_alive = [n for n in terminals["input_nodes"] if n in G]
        output_alive = [n for n in terminals["output_nodes"] if n in G]
        if not input_alive or not output_alive:
            raise RuntimeError(
                "Después del prune no quedan electrodos en el grafo vivo. "
                "La red probablemente no percola."
            )
        # Filtrar terminals por si algún electrodo quedó fuera
        terminals["input_nodes"] = input_alive
        terminals["output_nodes"] = output_alive
    return len(to_remove)


# ==============================================================================
# FUNCIONES DE PERCOLACIÓN Y CAMINOS
# ==============================================================================
def check_percolation(simulation: dict[str, Any]) -> bool:
    """Verifica si existe un camino topológico entre electrodos.

    Recorre las componentes conexas del grafo y comprueba si alguna
    contiene simultáneamente al menos un nodo de entrada y uno de salida.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"graph"`` y
        ``"terminals"``.

    Returns
    -------
    bool
        ``True`` si la red percola, ``False`` en caso contrario.

    Notes
    -----
    Complejidad :math:`O(V + E)`, muy inferior al chequeo por pares de
    electrodos.
    """
    G = simulation["graph"]
    input_set = set(simulation["terminals"]["input_nodes"])
    output_set = set(simulation["terminals"]["output_nodes"])

    if not input_set or not output_set:
        return False

    for component in nx.connected_components(G):
        if component & input_set and component & output_set:
            return True
    return False


def get_shortest_path_length(simulation: dict[str, Any]) -> int | None:
    """Longitud topológica del camino más corto entre electrodos.

    Calcula la cantidad mínima de saltos entre cualquier par
    ``(input_node, output_node)`` mediante un BFS multi-fuente: se
    inicializa la cola con todos los nodos de entrada simultáneamente y
    se corta al alcanzar el primer nodo de salida.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"graph"`` y
        ``"terminals"``.

    Returns
    -------
    int or None
        Número de saltos del camino más corto. ``None`` si no existe
        camino entre ningún par de electrodos.

    Notes
    -----
    Complejidad :math:`O(V + E)`.
    """
    from collections import deque

    G = simulation["graph"]
    input_nodes = simulation["terminals"]["input_nodes"]
    output_nodes = simulation["terminals"]["output_nodes"]

    if not input_nodes or not output_nodes:
        return None

    output_set = set(output_nodes)
    visited: dict = dict.fromkeys(input_nodes, 0)
    queue: deque = deque(input_nodes)

    while queue:
        node = queue.popleft()
        dist = visited[node]
        if node in output_set:
            return dist
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append(neighbor)

    return None
