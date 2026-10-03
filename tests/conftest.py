"""Shared pytest fixtures."""

from __future__ import annotations

import copy

import matplotlib

from neuromorphic.dinamica import run_simulation_dynamic_pulse

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pytest

from neuromorphic import setup_simulation


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Deterministic RNG for every test."""
    np.random.seed(42)
    yield


# ---------------------------------------------------------------------------
# Parameter dicts (stateless, safe to share across tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def small_params() -> dict:
    """Fast, non-percolating test network."""
    return {
        "NUM_WIRES": 200,
        "T_PULSE": 0.1,
        "T_RELAX": 0.2,
        "TIME_STEP_DT": 1e-2,
    }


@pytest.fixture(scope="session")
def percolating_params() -> dict:
    """Percolating test network, above the percolation threshold."""
    return {
        "NUM_WIRES": 1300,
        "T_PULSE": 0.05,
        "T_RELAX": 0.05,
        "TIME_STEP_DT": 1e-2,
    }


# ---------------------------------------------------------------------------
# Session-scoped expensive setups (built once, treated as read-only)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def _sim_small_base(small_params):
    return setup_simulation(parms=small_params)


@pytest.fixture(scope="session")
def _sim_small_unpruned(small_params):
    """Red chica sin podar, para tests estructurales del grafo."""
    return setup_simulation(parms=small_params, prune=False)


@pytest.fixture
def sim_small_unpruned(_sim_small_unpruned):
    return copy.deepcopy(_sim_small_unpruned)


@pytest.fixture(scope="session")
def _sim_percolating_base(percolating_params):
    return setup_simulation(parms=percolating_params)


# ---------------------------------------------------------------------------
# Function-scoped copies (so mutating tests don't leak state)
# ---------------------------------------------------------------------------
@pytest.fixture
def sim_small(_sim_small_base):
    return copy.deepcopy(_sim_small_base)


@pytest.fixture
def sim_percolating(_sim_percolating_base):
    return copy.deepcopy(_sim_percolating_base)


@pytest.fixture(scope="session")
def _pulse_run(_sim_percolating_base):
    """Run the pulse simulation once and share (sim, t, g, active).

    The simulation mutates the graph in place, so we run it on a deepcopy
    of the session-scoped base. The returned ``sim`` is the mutated copy;
    do NOT share it with tests that need a pristine graph.
    """
    sim = copy.deepcopy(_sim_percolating_base)
    t, g, active = run_simulation_dynamic_pulse(sim)
    return sim, t, g, active
