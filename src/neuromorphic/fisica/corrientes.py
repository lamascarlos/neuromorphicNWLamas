# corrientes.py
from typing import Any


# ==============================================================================
# CÁLCULOS DE CORRIENTES
# ==============================================================================
def _edge_conductance(data, p):
    """Conductancia de una arista: memristor (dinámica) o segmento resistivo."""
    if data.get("is_memristor", False):
        return data.get("conductance", p["G_OFF"])
    return 1.0 / (data["weight"] * p["R_WIRE_PER_LENGTH"] + 1e-12)


def calculate_input_current(simulation: dict[str, Any], V_solved) -> float:
    """Corriente neta inyectada por los electrodos de entrada.

    Suma las contribuciones :math:`(V_{\\text{in}} - V_{\\text{vecino}}) \\cdot G`
    sobre cada arista que conecta un nodo de entrada con un nodo fuera
    del conjunto de entradas.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"``, ``"terminals"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto.

    Returns
    -------
    float
        Corriente neta en amperios (positiva si entra a la red).
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    node_to_index = simulation["circuit"]["node_to_index"]
    input_nodes = simulation["terminals"]["input_nodes"]
    input_set = set(input_nodes)

    total = 0.0
    for in_node in input_nodes:
        V_in = V_solved[node_to_index[in_node]]
        for neighbor in G.neighbors(in_node):
            if neighbor in input_set:
                continue
            data = G.get_edge_data(in_node, neighbor)
            g = _edge_conductance(data, p)
            V_n = V_solved[node_to_index[neighbor]]
            total += (V_in - V_n) * g
    return total


def calculate_output_current(simulation: dict[str, Any], V_solved) -> float:
    """Corriente neta recolectada por los electrodos de salida.

    Suma las contribuciones :math:`(V_{\\text{vecino}} - V_{\\text{out}}) \\cdot G`
    sobre cada arista que conecta un nodo de salida con un nodo fuera
    del conjunto de salidas.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"``, ``"terminals"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto.

    Returns
    -------
    float
        Corriente neta en amperios.
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    node_to_index = simulation["circuit"]["node_to_index"]
    output_nodes = simulation["terminals"]["output_nodes"]
    output_set = set(output_nodes)

    total = 0.0
    for out_node in output_nodes:
        V_out = V_solved[node_to_index[out_node]]
        for neighbor in G.neighbors(out_node):
            if neighbor in output_set:
                continue
            data = G.get_edge_data(out_node, neighbor)
            g = _edge_conductance(data, p)
            V_n = V_solved[node_to_index[neighbor]]
            total += (V_n - V_out) * g
    return total
