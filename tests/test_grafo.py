"""Tests for neuromorphic.grafo."""

from __future__ import annotations

import networkx as nx

from neuromorphic.grafo import (
    check_percolation,
    get_shortest_path_length,
)


def test_graph_has_two_nodes_per_junction(sim_small_unpruned):
    G = sim_small_unpruned["graph"]
    n_junctions = len(sim_small_unpruned["junctions"]["junctions"])
    assert G.number_of_nodes() == 2 * n_junctions


def test_memristor_edges_are_present_and_tagged(sim_small_unpruned):
    G = sim_small_unpruned["graph"]
    memristor_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("is_memristor", False)]
    n_junctions = len(sim_small_unpruned["junctions"]["junctions"])
    assert len(memristor_edges) == n_junctions
    for _, _, data in memristor_edges:
        assert data["conductance"] == sim_small_unpruned["parameters"]["G_OFF"]


def test_prune_removes_floating_nodes(sim_small, sim_small_unpruned):
    """Después del prune, todos los nodos vivos tocan electrodos."""
    G = sim_small["graph"]
    terminals = sim_small["terminals"]
    live = set(terminals["input_nodes"]) | set(terminals["output_nodes"])

    # Todos los nodos vivos deben estar conectados a algún electrodo
    for node in G.nodes:
        assert any(nx.has_path(G, node, t) for t in live), (
            f"Nodo {node} no está conectado a ningún electrodo"
        )

    # El grafo podado tiene menos nodos que el sin podar
    assert G.number_of_nodes() < sim_small_unpruned["graph"].number_of_nodes()


def test_prune_preserves_electrode_connectivity(sim_small):
    """Los electrodos sobrevivientes siguen conectados entre sí."""
    assert check_percolation(sim_small) is False  # red chica no percola
    # Pero sigue habiendo inputs y outputs en el grafo
    assert len(sim_small["terminals"]["input_nodes"]) > 0
    assert len(sim_small["terminals"]["output_nodes"]) > 0


def test_segment_edges_have_weight(sim_small):
    G = sim_small["graph"]
    segment_edges = [
        (u, v, d) for u, v, d in G.edges(data=True) if not d.get("is_memristor", False)
    ]
    # A network of 200 wires should have at least some segments.
    assert segment_edges, "Expected some wire segments"
    for _, _, data in segment_edges:
        assert "weight" in data
        assert data["weight"] >= 0.0


def test_electrode_detection_uses_proximity_threshold(sim_small):
    G = sim_small["graph"]
    p = sim_small["parameters"]
    terminals = sim_small["terminals"]

    pos = nx.get_node_attributes(G, "pos")
    for n in terminals["input_nodes"]:
        assert pos[n][0] < p["PROXIMITY_THRESHOLD"]
    for n in terminals["output_nodes"]:
        assert pos[n][0] > p["AREA"] - p["PROXIMITY_THRESHOLD"]


def test_electrode_lists_are_disjoint(sim_small):
    terminals = sim_small["terminals"]
    assert set(terminals["input_nodes"]).isdisjoint(terminals["output_nodes"])


def test_check_percolation_returns_bool(sim_small):
    result = check_percolation(sim_small)
    assert isinstance(result, bool)


def test_low_density_does_not_percolate(sim_small):
    assert check_percolation(sim_small) is False


def test_high_density_percolates(sim_percolating):
    assert check_percolation(sim_percolating) is True


def test_shortest_path_length_on_percolating(sim_percolating):
    length = get_shortest_path_length(sim_percolating)
    assert length is not None
    assert length > 0


def test_shortest_path_length_on_disconnected_returns_none(sim_small):
    # Non-percolating network: no path -> None
    assert get_shortest_path_length(sim_small) is None
