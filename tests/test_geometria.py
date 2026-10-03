"""Tests for neuromorphic.geometria."""

from __future__ import annotations

import numpy as np
import pytest

from neuromorphic.geometria import generate_and_find_junctions


def test_wire_count_matches_config(sim_small):
    wires = sim_small["junctions"]["wires"]
    assert len(wires) == sim_small["parameters"]["NUM_WIRES"]


def test_wire_structure(sim_small):
    wires = sim_small["junctions"]["wires"]
    for w in wires[:5]:
        assert set(w.keys()) == {"id", "p1", "p2"}
        assert isinstance(w["p1"], np.ndarray)
        assert isinstance(w["p2"], np.ndarray)
        assert w["p1"].shape == (2,)
        assert w["p2"].shape == (2,)


def test_wire_length_is_constant(sim_small):
    p = sim_small["parameters"]
    expected = p["LENGTH"]
    wires = sim_small["junctions"]["wires"]
    for w in wires[:10]:
        length = np.linalg.norm(w["p2"] - w["p1"])
        assert length == pytest.approx(expected, rel=1e-9)


def test_wire_ids_are_unique_and_dense(sim_small):
    wires = sim_small["junctions"]["wires"]
    ids = [w["id"] for w in wires]
    assert ids == list(range(len(wires)))


def test_junctions_have_expected_structure(sim_small):
    junctions = sim_small["junctions"]["junctions"]
    for j in junctions[:5]:
        assert set(j.keys()) == {"id", "pos", "wires"}
        assert isinstance(j["pos"], np.ndarray)
        assert j["pos"].shape == (2,)
        assert isinstance(j["wires"], tuple)
        assert len(j["wires"]) == 2
        w1, w2 = j["wires"]
        assert w1 != w2


def test_wire_to_junctions_maps_all_wires(sim_small):
    w2j = sim_small["junctions"]["wire_to_junctions"]
    num_wires = sim_small["parameters"]["NUM_WIRES"]
    assert set(w2j.keys()) == set(range(num_wires))
    # Every junction must appear in the adjacency of both its wires.
    for j in sim_small["junctions"]["junctions"]:
        w1, w2 = j["wires"]
        assert j in w2j[w1]
        assert j in w2j[w2]


def test_low_density_produces_few_junctions(sim_small):
    """With NUM_WIRES=200 the network is well below percolation."""
    junctions = sim_small["junctions"]["junctions"]
    # Empirical: a handful to low-hundreds. Bound generously.
    assert len(junctions) < 200


def test_higher_density_produces_more_junctions(sim_small):
    """Doubling the wire count should increase junction density."""
    from neuromorphic.load_config import load_parameters

    sim = load_parameters(parms={"NUM_WIRES": 400})
    generate_and_find_junctions(sim)
    n_200 = len(sim_small["junctions"]["junctions"])
    n_400 = len(sim["junctions"]["junctions"])
    assert n_400 > n_200
