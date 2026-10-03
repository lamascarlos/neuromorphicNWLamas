"""Tests específicos del evolver ``stochastic2`` (modelo de Lamas et al., 2026)."""

import numpy as np
import pytest

from neuromorphic.fisica import EVOLVER_SPECS

stochastic2 = EVOLVER_SPECS["stochastic2"].update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_state(sim, g_value: float) -> None:
    """Fija todos los memristores (array y espejo del grafo) en ``g_value``."""
    circuit = sim["circuit"]
    circuit["memristor_g"][:] = g_value
    for u, v in circuit["mem_edge_keys"]:
        sim["graph"].edges[u, v]["conductance"] = g_value


def _voltages_with_vmem(sim, v_mem: float) -> np.ndarray:
    """Vector nodal con caída ``v_mem`` sobre cada memristor.

    Cada nodo del grafo pertenece a exactamente un memristor (topología
    por duplicación de nodos), así que basta con fijar un extremo en
    ``v_mem`` y el otro en 0.
    """
    circuit = sim["circuit"]
    V = np.zeros(circuit["N"])
    V[circuit["mem_u_idx"]] = v_mem
    return V


def _assert_binomial(n_switched: int, n: int, p: float, n_sigma: float = 5.0) -> None:
    sigma = np.sqrt(n * p * (1 - p))
    assert abs(n_switched - n * p) < n_sigma * sigma, (n_switched, n * p, sigma)


def test_stochastic2_is_registered():
    assert "stochastic2" in EVOLVER_SPECS


def test_each_node_belongs_to_one_memristor(sim_percolating):
    """Precondición de ``_voltages_with_vmem``."""
    c = sim_percolating["circuit"]
    idx = np.concatenate([c["mem_u_idx"], c["mem_v_idx"]])
    assert len(np.unique(idx)) == len(idx)


# ---------------------------------------------------------------------------
# SET
# ---------------------------------------------------------------------------
def test_no_set_below_threshold(sim_percolating):
    p = sim_percolating["parameters"]
    _set_state(sim_percolating, p["G_OFF"])
    V = _voltages_with_vmem(sim_percolating, 0.5 * p["V_THRESHOLD"])
    p["P0_SET"] = 1e12  # incluso con tasa enorme, sin sobretensión no hay SET

    stochastic2(sim_percolating, V)

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_OFF"])


def test_set_rate_matches_formula(sim_percolating):
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 1.0
    overdrive = 1.0
    _set_state(sim_percolating, p["G_OFF"])
    V = _voltages_with_vmem(sim_percolating, p["V_THRESHOLD"] + overdrive)

    stochastic2(sim_percolating, V)

    mem_g = sim_percolating["circuit"]["memristor_g"]
    expected = p["TIME_STEP_DT"] * p["P0_SET"] * (1 - np.exp(-p["ALPHA_SET"] * overdrive))
    _assert_binomial(int((mem_g == p["G_ON"]).sum()), mem_g.size, expected)


def test_set_probability_scales_with_dt(sim_percolating):
    """Con Δt diez veces menor, la tasa de SET por paso cae diez veces."""
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 0.1
    p["P0_SET"] = 5.0
    _set_state(sim_percolating, p["G_OFF"])
    V = _voltages_with_vmem(sim_percolating, 1e3)  # saturación: 1 - exp(-alpha * overdrive) ~ 1

    stochastic2(sim_percolating, V)

    mem_g = sim_percolating["circuit"]["memristor_g"]
    _assert_binomial(int((mem_g == p["G_ON"]).sum()), mem_g.size, 0.5)


# ---------------------------------------------------------------------------
# RESET
# ---------------------------------------------------------------------------
def test_no_reset_without_current(sim_percolating):
    p = sim_percolating["parameters"]
    p["P_DECAY"] = 1e30
    _set_state(sim_percolating, p["G_ON"])

    stochastic2(sim_percolating, np.zeros(sim_percolating["circuit"]["N"]))

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_ON"])


def test_reset_rate_matches_formula(sim_percolating):
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 1.0
    v_mem = 1.0
    i_mem = p["G_ON"] * v_mem
    target = 0.2
    p["P_DECAY"] = target / (p["TIME_STEP_DT"] * i_mem**2)
    _set_state(sim_percolating, p["G_ON"])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, v_mem))

    mem_g = sim_percolating["circuit"]["memristor_g"]
    _assert_binomial(int((mem_g == p["G_OFF"]).sum()), mem_g.size, target)


def test_reset_probability_is_capped_at_one(sim_percolating):
    p = sim_percolating["parameters"]
    p["P_DECAY"] = 1e30
    _set_state(sim_percolating, p["G_ON"])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_OFF"])


# ---------------------------------------------------------------------------
# Consistencia
# ---------------------------------------------------------------------------
def test_no_set_and_reset_in_same_step(sim_percolating):
    """Una juntura que hace SET no puede hacer RESET en el mismo paso."""
    p = sim_percolating["parameters"]
    p["P0_SET"] = 1e30
    p["P_DECAY"] = 1e30
    _set_state(sim_percolating, p["G_OFF"])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_ON"])


@pytest.mark.parametrize("initial", ["G_OFF", "G_ON"])
def test_graph_mirrors_memristor_array(sim_percolating, initial):
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 1.0
    p["P_DECAY"] = 0.3 / p["G_ON"] ** 2
    _set_state(sim_percolating, p[initial])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    circuit = sim_percolating["circuit"]
    G = sim_percolating["graph"]
    for j, (u, v) in enumerate(circuit["mem_edge_keys"]):
        assert G.edges[u, v]["conductance"] == circuit["memristor_g"][j]


def test_full_run_with_stochastic2(sim_percolating):
    from neuromorphic.dinamica import run_simulation_dynamic_pulse

    sim_percolating["parameters"]["EVOLVER"] = "stochastic2"
    t, g, active = run_simulation_dynamic_pulse(sim_percolating)

    assert len(t) == len(g) == len(active) > 0
    assert np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# Registro: init, parámetros propios y máscara de activos
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_model():
    """Registra un modelo temporal y lo borra al terminar el test."""
    from neuromorphic.fisica import EVOLVER_SPECS

    names: list[str] = []
    yield names
    for name in names:
        EVOLVER_SPECS.pop(name, None)


def test_custom_init_and_parameter_defaults(sim_percolating, tmp_model):
    from neuromorphic.fisica import (
        initialize_evolver,
        register_memristor_evol_model,
        set_all_memristors,
    )

    def init_on(simulation):
        set_all_memristors(simulation, simulation["parameters"]["G_ON"])
        simulation["evolver_state"]["rate"] = simulation["parameters"]["_TMP_RATE"]

    @register_memristor_evol_model("_tmp_on", init=init_on, parameters={"_TMP_RATE": 3.0})
    def _tmp(simulation, V_solved):
        return simulation["graph"]

    tmp_model.append("_tmp_on")

    p = sim_percolating["parameters"]
    p["EVOLVER"] = "_tmp_on"
    initialize_evolver(sim_percolating)

    assert p["_TMP_RATE"] == 3.0  # default completado
    assert sim_percolating["evolver_state"] == {"rate": 3.0}
    assert sim_percolating["circuit"]["memristor_active"].all()

    p["_TMP_RATE"] = 7.0  # un valor ya presente no se pisa
    initialize_evolver(sim_percolating)
    assert sim_percolating["evolver_state"]["rate"] == 7.0


def test_unknown_evolver_raises(sim_percolating):
    from neuromorphic.fisica import initialize_evolver

    sim_percolating["parameters"]["EVOLVER"] = "no_existe"
    with pytest.raises(KeyError, match="no_existe"):
        initialize_evolver(sim_percolating)


def test_setup_leaves_evolver_initialized(sim_percolating):
    circuit = sim_percolating["circuit"]
    assert "evolver_state" in sim_percolating
    assert circuit["memristor_active"].shape == circuit["memristor_g"].shape
    assert not circuit["memristor_active"].any()


@pytest.mark.parametrize("name", ["stochastic1", "stochastic2"])
def test_binary_models_keep_mask_in_sync(sim_percolating, name):
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 1.0
    _set_state(sim_percolating, p["G_OFF"])

    EVOLVER_SPECS[name].update(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    circuit = sim_percolating["circuit"]
    assert circuit["memristor_active"].any()
    np.testing.assert_array_equal(circuit["memristor_active"], circuit["memristor_g"] == p["G_ON"])


def test_rerun_starts_from_initial_state(sim_percolating, capsys):
    """Correr dos veces la misma simulación parte del mismo estado inicial."""
    from neuromorphic.dinamica import run_simulation_dynamic_pulse

    sim_percolating["parameters"]["T_RELAX"] = 0.0  # terminar en pleno pulso
    _, g1, a1 = run_simulation_dynamic_pulse(sim_percolating)
    assert sim_percolating["circuit"]["memristor_active"].any()  # quedó estado ON
    _, g2, a2 = run_simulation_dynamic_pulse(sim_percolating)

    assert a2[0] == a1[0] == 0
    assert g2[0] == pytest.approx(g1[0])


@pytest.mark.parametrize(
    "module", ["neuromorphic", "neuromorphic.fisica", "neuromorphic.fisica.evolvers"]
)
def test_all_exports_exist(module):
    """Cada nombre de ``__all__`` existe (``from module import *`` no falla)."""
    import importlib

    mod = importlib.import_module(module)
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert missing == []


# ---------------------------------------------------------------------------
# Descubrimiento automático de módulos
# ---------------------------------------------------------------------------
def test_all_public_model_modules_are_discovered():
    import sys
    from pathlib import Path

    import neuromorphic.fisica.evolvers as evolvers_pkg

    pkg_dir = Path(evolvers_pkg.__file__).parent
    expected = sorted(
        f.stem for f in pkg_dir.glob("*.py") if not f.stem.startswith("_") and f.stem != "base"
    )
    assert expected == evolvers_pkg.MODEL_MODULES
    for name in expected:
        assert f"neuromorphic.fisica.evolvers.{name}" in sys.modules


def test_private_modules_are_not_imported():
    """Importar el paquete no carga módulos privados ni dependencias opcionales."""
    import subprocess
    import sys

    code = (
        "import sys, neuromorphic; "
        "assert 'numba' not in sys.modules; "
        "assert 'neuromorphic.fisica.evolvers._ladder_numba_kernel' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
