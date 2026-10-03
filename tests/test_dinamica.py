"""Tests for neuromorphic.simulador."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from neuromorphic.dinamica import run_simulation_dynamic_pulse


# ---------------------------------------------------------------------------
# Happy path: use the session-cached run
# ---------------------------------------------------------------------------
def test_run_returns_three_histories(_pulse_run):
    _, t, g, act = _pulse_run
    assert len(t) == len(g) == len(act)
    assert len(t) > 0


def test_history_length_matches_total_steps(_pulse_run, percolating_params):
    p = percolating_params
    # Debe coincidir con lo que hace simulador.py:
    #   total_steps = int((T_PULSE + T_RELAX) / TIME_STEP_DT)
    expected = int((p["T_PULSE"] + p["T_RELAX"]) / p["TIME_STEP_DT"])
    _, t, _, _ = _pulse_run
    assert len(t) == expected


def test_time_is_monotonic(_pulse_run):
    _, t, _, _ = _pulse_run
    for prev, nxt in pairwise((t, t[1:])):
        assert nxt > prev


def test_active_count_is_non_negative(_pulse_run):
    _, _, _, act = _pulse_run
    assert all(a >= 0 for a in act)


def test_all_histories_are_finite(_pulse_run):
    _, t, g, act = _pulse_run
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(g))
    assert np.all(np.isfinite(act))


# ---------------------------------------------------------------------------
# Error path: build a fresh small network and verify the rejection
# ---------------------------------------------------------------------------
def test_non_percolating_raises(sim_small):
    """A network below threshold must be rejected explicitly."""
    with pytest.raises(RuntimeError):
        run_simulation_dynamic_pulse(sim_small)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------
STEP_DATA_KEYS = {"Y", "I_vec", "V_vec", "V_input", "G_total"}


def test_callback_is_called_once_per_step(sim_percolating):
    calls = []

    def cb(t, simulation, step_data):
        calls.append((t, simulation, set(step_data)))

    t_hist, _, _ = run_simulation_dynamic_pulse(sim_percolating, callback=cb)

    assert [c[0] for c in calls] == t_hist
    assert all(c[1] is sim_percolating for c in calls)
    assert all(keys == STEP_DATA_KEYS for _, _, keys in calls)


def test_callback_step_data_matches_history(sim_percolating):
    p = sim_percolating["parameters"]
    v_inputs, g_totals = [], []

    def cb(t, simulation, step_data):
        v_inputs.append((t, step_data["V_input"]))
        g_totals.append(step_data["G_total"])

    _, g_hist, _ = run_simulation_dynamic_pulse(sim_percolating, callback=cb)

    assert g_totals == g_hist
    for t, v in v_inputs:
        assert v == (p["V_INPUT"] if t <= p["T_PULSE"] else p["V_READ"])


def test_callback_step_data_solves_circuit(sim_percolating):
    residuals = []

    def cb(t, simulation, step_data):
        Y, I_vec, V_vec = step_data["Y"], step_data["I_vec"], step_data["V_vec"]
        residuals.append(np.abs(Y @ V_vec - I_vec).max() / np.abs(I_vec).max())

    run_simulation_dynamic_pulse(sim_percolating, callback=cb)
    assert max(residuals) < 1e-8


def test_callback_exception_interrupts_run(sim_percolating):
    class StopRunError(Exception):
        pass

    seen = []

    def cb(t, simulation, step_data):
        seen.append(t)
        if len(seen) == 3:
            raise StopRunError

    with pytest.raises(StopRunError):
        run_simulation_dynamic_pulse(sim_percolating, callback=cb)
    assert len(seen) == 3
