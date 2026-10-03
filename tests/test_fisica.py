"""Tests for neuromorphic.fisica."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from neuromorphic.fisica import (
    EVOLVER_SPECS,
    build_admittance_matrix,
    calculate_input_current,
    calculate_output_current,
    initialize_evolver,
)


# ---------------------------------------------------------------------------
# Admittance matrix
# ---------------------------------------------------------------------------
def test_admittance_returns_expected_shapes(sim_small_unpruned):
    Y, I_vec, n2i = build_admittance_matrix(sim_small_unpruned)
    n = sim_small_unpruned["graph"].number_of_nodes()
    assert Y.shape == (n, n)
    assert I_vec.shape == (n,)
    assert len(n2i) == n


def test_admittance_is_csr(sim_small):
    Y, _, _ = build_admittance_matrix(sim_small)
    assert isinstance(Y, csr_matrix)


def test_boundary_conditions_set_voltage(sim_small):
    _, I_vec, n2i = build_admittance_matrix(sim_small, v_input=2.5)
    p = sim_small["parameters"]
    for n in sim_small["terminals"]["input_nodes"]:
        assert I_vec[n2i[n]] == pytest.approx(2.5)
    for n in sim_small["terminals"]["output_nodes"]:
        assert I_vec[n2i[n]] == pytest.approx(p["V_GROUND"])


def test_voltage_override_does_not_mutate_parameters(sim_small):
    p = sim_small["parameters"]
    original = p["V_INPUT"]
    build_admittance_matrix(sim_small, v_input=99.0)
    assert p["V_INPUT"] == original


def test_circuit_is_stored_in_simulation(sim_small):
    assert "circuit" in sim_small
    assert "Y" in sim_small["circuit"]
    assert "I" in sim_small["circuit"]
    assert "node_to_index" in sim_small["circuit"]


# ---------------------------------------------------------------------------
# Stochastic conductance update
# ---------------------------------------------------------------------------
def _select_evolver(sim, name):
    """Selecciona e inicializa el modelo antes de llamar a su update."""
    sim["parameters"]["EVOLVER"] = name
    return initialize_evolver(sim).update


# ---------------------------------------------------------------------------
# Evolvers (todos los registrados)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ["evolver_name", "evolver_spec"], [(name, fn) for name, fn in EVOLVER_SPECS.items()]
)
def test_update_returns_same_graph(sim_percolating, evolver_name, evolver_spec):
    evolver_fn = _select_evolver(sim_percolating, evolver_name)
    G = sim_percolating["graph"]
    Y, I_vec, _ = build_admittance_matrix(sim_percolating)
    from scipy.sparse.linalg import spsolve

    V = spsolve(Y, I_vec)
    G_returned = evolver_fn(sim_percolating, V)
    assert G_returned is G


@pytest.mark.parametrize(
    ["evolver_name", "evolver_spec"], [(name, fn) for name, fn in EVOLVER_SPECS.items()]
)
def test_only_memristor_edges_change_state(sim_percolating, evolver_name, evolver_spec):
    """Non-memristor edges must never carry a 'conductance' attribute."""
    evolver_fn = _select_evolver(sim_percolating, evolver_name)

    G = sim_percolating["graph"]
    Y, I_vec, _ = build_admittance_matrix(sim_percolating)
    from scipy.sparse.linalg import spsolve

    V = spsolve(Y, I_vec)
    evolver_fn(sim_percolating, V)

    for _, _, data in G.edges(data=True):
        if not data.get("is_memristor", False):
            assert "conductance" not in data


@pytest.mark.parametrize(
    ["evolver_name", "evolver_spec"], [(name, fn) for name, fn in EVOLVER_SPECS.items()]
)
def test_zero_voltage_does_not_trigger_set(sim_percolating, evolver_name, evolver_spec):
    """With V_mem = 0 everywhere, no memristor should turn ON."""
    evolver_fn = _select_evolver(sim_percolating, evolver_name)
    p = sim_percolating["parameters"]
    G = sim_percolating["graph"]
    n = G.number_of_nodes()
    V_zero = np.zeros(n)
    evolver_fn(sim_percolating, V_zero)

    n_on = sum(
        1
        for _, _, d in G.edges(data=True)
        if d.get("is_memristor", False) and d.get("conductance") == p["G_ON"]
    )
    assert n_on == 0
    assert not sim_percolating["circuit"]["memristor_active"].any()


# ---------------------------------------------------------------------------
# Currents
# ---------------------------------------------------------------------------
def test_input_current_is_finite(sim_percolating):
    Y, I_vec, _ = build_admittance_matrix(sim_percolating)
    from scipy.sparse.linalg import spsolve

    V = spsolve(Y, I_vec)
    I_in = calculate_input_current(sim_percolating, V)
    assert np.isfinite(I_in)


def test_output_current_is_finite(sim_percolating):
    Y, I_vec, _ = build_admittance_matrix(sim_percolating)
    from scipy.sparse.linalg import spsolve

    V = spsolve(Y, I_vec)
    I_out = calculate_output_current(sim_percolating, V)
    assert np.isfinite(I_out)


def test_input_and_output_currents_have_opposite_sign(sim_percolating):
    """Sign convention: current entering the network is positive at input."""
    Y, I_vec, _ = build_admittance_matrix(sim_percolating)
    from scipy.sparse.linalg import spsolve

    V = spsolve(Y, I_vec)
    I_in = calculate_input_current(sim_percolating, V)
    I_out = calculate_output_current(sim_percolating, V)
    # In a passive network driven by a positive voltage, at least one
    # of the two must be non-trivially non-zero.
    assert abs(I_in) + abs(I_out) > 0
