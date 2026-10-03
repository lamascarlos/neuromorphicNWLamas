"""Tests de los modelos de junturas Ag/PVP/Ag: ``ladder``, ``ladder_numba`` y ``thermal``."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from neuromorphic.fisica import EVOLVER_SPECS, initialize_evolver
from neuromorphic.fisica.evolvers._junction import G0, filament_closed, make_ladder
from neuromorphic.fisica.evolvers.thermal import StepSaturationWarning, thermal_conductance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _select(sim, name, **params):
    sim["parameters"]["EVOLVER"] = name
    sim["parameters"].update(params)
    initialize_evolver(sim)
    return EVOLVER_SPECS[name].update


def _voltages_with_vmem(sim, v_mem) -> np.ndarray:
    """Caída ``v_mem`` (escalar o array por juntura) sobre cada memristor."""
    circuit = sim["circuit"]
    V = np.zeros(circuit["N"])
    V[circuit["mem_u_idx"]] = v_mem
    return V


def _assert_consistent(sim):
    """Array, espejo del grafo y máscara de activos coherentes."""
    circuit = sim["circuit"]
    G = sim["graph"]
    for j, (u, v) in enumerate(circuit["mem_edge_keys"]):
        assert G.edges[u, v]["conductance"] == circuit["memristor_g"][j]
    np.testing.assert_array_equal(
        circuit["memristor_active"], filament_closed(circuit["memristor_g"])
    )


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


needs_numba = pytest.mark.skipif(not _numba_available(), reason="numba no instalado")
LADDER_MODELS = ["ladder", pytest.param("ladder_numba", marks=needs_numba)]


# ---------------------------------------------------------------------------
# Escalera
# ---------------------------------------------------------------------------
def test_make_ladder_levels():
    ladder, gfac, edges = make_ladder(1e-9, 2.5e-10, 5e9, 10)
    assert ladder.size == 4 + 10  # 4 niveles túnel + 10 metálicos
    assert np.all(np.diff(ladder) > 0)
    assert ladder[0] == pytest.approx(G0 * np.exp(-10.0))
    assert ladder[4] == pytest.approx(G0)
    assert np.all((edges > ladder[:-1]) & (edges < ladder[1:]))
    assert gfac[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ladder / ladder_numba
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", LADDER_MODELS)
def test_ladder_init(sim_percolating, name):
    _select(sim_percolating, name)
    ladder = sim_percolating["evolver_state"]["ladder"]
    assert np.all(sim_percolating["circuit"]["memristor_g"] == ladder[0])
    assert not sim_percolating["circuit"]["memristor_active"].any()
    _assert_consistent(sim_percolating)


@pytest.mark.parametrize("name", LADDER_MODELS)
def test_ladder_stays_on_levels_and_consistent(sim_percolating, name):
    update = _select(sim_percolating, name, TIME_STEP_DT=1e-3)
    V = _voltages_with_vmem(sim_percolating, 1.0)
    for _ in range(20):
        update(sim_percolating, V)

    ladder = sim_percolating["evolver_state"]["ladder"]
    mem_g = sim_percolating["circuit"]["memristor_g"]
    assert np.isin(mem_g, ladder).all()
    assert sim_percolating["circuit"]["memristor_active"].any()  # algunos cerraron el gap
    _assert_consistent(sim_percolating)


@pytest.mark.parametrize("name", LADDER_MODELS)
def test_ladder_zero_rate_junctions_do_not_grow(sim_percolating, name):
    n = sim_percolating["circuit"]["memristor_g"].size
    nu_up = np.where(np.arange(n) % 2 == 0, 0.0, 1e3)
    update = _select(sim_percolating, name, LADDER_NU_UP=nu_up, TIME_STEP_DT=1e-3)
    V = _voltages_with_vmem(sim_percolating, 1.0)
    for _ in range(10):
        update(sim_percolating, V)

    mem_g = sim_percolating["circuit"]["memristor_g"]
    ladder = sim_percolating["evolver_state"]["ladder"]
    assert np.all(mem_g[nu_up == 0.0] == ladder[0])
    assert np.any(mem_g[nu_up > 0.0] > ladder[0])


@pytest.mark.parametrize("name", LADDER_MODELS)
def test_ladder_rupture_resets_to_bottom(sim_percolating, name):
    update = _select(sim_percolating, name, LADDER_NU_RP=1e30, LADDER_NU_UP=0.0, LADDER_NU_DN=0.0)
    ladder = sim_percolating["evolver_state"]["ladder"]
    from neuromorphic.fisica import set_all_memristors

    set_all_memristors(sim_percolating, ladder[-1])
    update(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    assert np.all(sim_percolating["circuit"]["memristor_g"] == ladder[0])
    _assert_consistent(sim_percolating)


def test_ladder_geometry_must_be_scalar(sim_percolating):
    n = sim_percolating["circuit"]["memristor_g"].size
    with pytest.raises(ValueError, match="GAP_D"):
        _select(sim_percolating, "ladder", GAP_D=np.full(n, 1e-9))


@pytest.mark.parametrize("name", LADDER_MODELS)
def test_ladder_run_is_reproducible(sim_percolating, name):
    sim_percolating["parameters"]["EVOLVER"] = name
    from neuromorphic.dinamica import run_simulation_dynamic_pulse

    _, g1, a1 = run_simulation_dynamic_pulse(copy.deepcopy(sim_percolating))
    _, g2, a2 = run_simulation_dynamic_pulse(copy.deepcopy(sim_percolating))
    assert a1 == a2
    np.testing.assert_allclose(g1, g2)


@needs_numba
def test_numba_matches_numpy_statistically(sim_percolating):
    """Sin truncamiento relevante, ambos kernels dan la misma distribución de saltos."""
    params = {
        "TIME_STEP_DT": 1e-4,
        "LADDER_NU_DN": 0.0,
        "LADDER_NU_RP": 0.0,
        "LADDER_MAX_JUMPS": 50,
    }
    v_mem = 0.3  # r_up·dt ~ 0.4 en el nivel más bajo
    means = {}
    for name in ["ladder", "ladder_numba"]:
        sim = copy.deepcopy(sim_percolating)
        update = _select(sim, name, **params)
        update(sim, _voltages_with_vmem(sim, v_mem))
        k = np.searchsorted(sim["evolver_state"]["edges"], sim["circuit"]["memristor_g"])
        means[name] = (k.mean(), k.std() / np.sqrt(k.size))
    (m1, e1), (m2, e2) = means["ladder"], means["ladder_numba"]
    assert abs(m1 - m2) < 5 * np.hypot(e1, e2)


# ---------------------------------------------------------------------------
# thermal
# ---------------------------------------------------------------------------
def test_thermal_conductance_is_monotonic_and_continuous():
    from neuromorphic.fisica.evolvers.thermal import THERMAL_PARAMETERS

    p = dict(THERMAL_PARAMETERS)
    lam = np.linspace(0.0, p["THERMAL_LAM_MAX"], 2001)
    g = thermal_conductance(lam, p)
    assert np.all(np.diff(g) >= 0)
    assert g[0] == pytest.approx(G0 * np.exp(-2 * p["KAPPA"] * p["GAP_D"]))
    assert thermal_conductance(np.array([1.0]), p)[0] == pytest.approx(G0)
    assert thermal_conductance(np.array([1.0 - 1e-9]), p)[0] == pytest.approx(G0, rel=1e-6)
    assert g[-1] == pytest.approx(p["N_MAX"] * G0)


def test_thermal_init(sim_percolating):
    _select(sim_percolating, "thermal")
    state = sim_percolating["evolver_state"]
    assert np.all(state["lam"] == 0.0)
    assert not sim_percolating["circuit"]["memristor_active"].any()
    _assert_consistent(sim_percolating)


def test_thermal_grows_under_bias_and_decays_without(sim_percolating):
    update = _select(sim_percolating, "thermal", TIME_STEP_DT=1e-4)
    for _ in range(50):
        update(sim_percolating, _voltages_with_vmem(sim_percolating, 0.8))
    lam_biased = sim_percolating["evolver_state"]["lam"].copy()
    assert np.all(lam_biased > 0.0)

    update(sim_percolating, _voltages_with_vmem(sim_percolating, 0.0))
    lam_relaxed = sim_percolating["evolver_state"]["lam"]
    assert np.all(lam_relaxed <= lam_biased)
    assert np.any(lam_relaxed < lam_biased)
    _assert_consistent(sim_percolating)


def test_thermal_rupture(sim_percolating):
    update = _select(sim_percolating, "thermal", THERMAL_T_RUPT=300.0, TIME_STEP_DT=1e-4)
    sim_percolating["evolver_state"]["lam"][:] = 2.0
    update(sim_percolating, _voltages_with_vmem(sim_percolating, 0.1))  # T > 300 K
    state = sim_percolating["evolver_state"]
    assert state["ruptured"].all()
    assert np.all(state["lam"] == 0.0)


def test_thermal_saturation_warns_once(sim_percolating):
    update = _select(sim_percolating, "thermal", TIME_STEP_DT=1e-2)
    V = _voltages_with_vmem(sim_percolating, 1.0)
    with pytest.warns(StepSaturationWarning):
        update(sim_percolating, V)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StepSaturationWarning)
        update(sim_percolating, V)  # segunda vez: silencio
    assert sim_percolating["evolver_state"]["saturated"].any()


def test_thermal_per_junction_parameters(sim_percolating):
    n = sim_percolating["circuit"]["memristor_g"].size
    ea = np.where(np.arange(n) % 2 == 0, 5.0, 0.6)  # barrera enorme en la mitad
    update = _select(sim_percolating, "thermal", THERMAL_EA=ea, TIME_STEP_DT=1e-4)
    for _ in range(20):
        update(sim_percolating, _voltages_with_vmem(sim_percolating, 0.8))
    lam = sim_percolating["evolver_state"]["lam"]
    assert lam[ea == 5.0].max() < 1e-6 < lam[ea == 0.6].min()


# ---------------------------------------------------------------------------
# Corrida completa
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["ladder", pytest.param("ladder_numba", marks=needs_numba), "thermal"]
)
def test_full_run(sim_percolating, name):
    from neuromorphic.dinamica import run_simulation_dynamic_pulse

    sim_percolating["parameters"].update(EVOLVER=name, TIME_STEP_DT=1e-3)
    t, g, active = run_simulation_dynamic_pulse(sim_percolating)
    assert len(t) == len(g) == len(active) > 0
    assert np.all(np.isfinite(g))
    _assert_consistent(sim_percolating)
