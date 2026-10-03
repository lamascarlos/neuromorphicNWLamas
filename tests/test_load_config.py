"""Tests for neuromorphic.load_config."""

from __future__ import annotations

import warnings

import pytest

from neuromorphic.load_config import (
    DERIVED_KEYS,
    KNOWN_RAW_KEYS,
    UnknownParameterWarning,
    load_parameters,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_defaults_load():
    """Loading without arguments uses packaged defaults."""
    result = load_parameters()
    p = result["parameters"]
    assert p["NUM_WIRES"] > 0
    assert p["AREA"] > 0
    assert p["LENGTH"] > 0
    assert p["T_PULSE"] > 0
    assert p["T_RELAX"] > 0


def test_all_known_keys_present_in_defaults():
    """Every key in KNOWN_RAW_KEYS must appear in the loaded defaults."""
    params = load_parameters()["parameters"]
    missing = KNOWN_RAW_KEYS - params.keys()
    assert not missing, f"Known keys missing from defaults: {missing}"


def test_all_derived_keys_present():
    """Every derived key must be computed."""
    params = load_parameters()["parameters"]
    missing = DERIVED_KEYS - params.keys()
    assert not missing, f"Derived keys missing: {missing}"


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------
def test_override_applied():
    result = load_parameters(parms={"NUM_WIRES": 42})
    assert result["parameters"]["NUM_WIRES"] == 42


def test_derived_recomputed_after_area_override():
    result = load_parameters(parms={"AREA": 500.0})
    p = result["parameters"]
    assert p["PROXIMITY_THRESHOLD"] == p["PROXIMITY_THRESHOLD_RATIO"] * 500.0


def test_derived_recomputed_after_diameter_override():
    import numpy as np

    result = load_parameters(parms={"DIAMETRO_NM": 50.0})
    p = result["parameters"]
    radio_um = (50.0 / 2.0) * 1e-3
    expected = p["RHO_PLATA"] / (np.pi * radio_um**2)
    assert p["R_WIRE_PER_LENGTH"] == pytest.approx(expected)


def test_total_time_is_sum_of_pulse_and_relax():
    result = load_parameters(parms={"T_PULSE": 3.0, "T_RELAX": 7.0})
    assert result["parameters"]["TOTAL_TIME"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------
def test_unknown_key_warns():
    with pytest.warns(UnknownParameterWarning, match="NUM_WIRE"):
        result = load_parameters(parms={"NUM_WIRE": 100})  # typo intentional
    # The typo key does not override the real one:
    assert result["parameters"]["NUM_WIRES"] != 100


def test_unknown_key_is_kept_alongside_valid_one():
    with pytest.warns(UnknownParameterWarning):
        result = load_parameters(parms={"NUM_WIRES": 42, "MY_EXTRA": 999})
    p = result["parameters"]
    assert p["NUM_WIRES"] == 42
    assert p["MY_EXTRA"] == 999


def test_declared_evolver_parameter_does_not_warn():
    from neuromorphic.fisica.evolvers import EVOLVER_SPECS, register_memristor_evol_model

    @register_memristor_evol_model("_tmp_declared", parameters={"_TMP_RATE": 1.0})
    def _tmp(simulation, V_solved):
        return simulation["graph"]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p = load_parameters(parms={"_TMP_RATE": 2.5})["parameters"]
        assert p["_TMP_RATE"] == 2.5
    finally:
        EVOLVER_SPECS.pop("_tmp_declared")


def test_array_parameter_is_accepted():
    import numpy as np

    with pytest.warns(UnknownParameterWarning):
        p = load_parameters(parms={"PER_JUNCTION": np.arange(3.0)})["parameters"]
    assert p["PER_JUNCTION"].tolist() == [0.0, 1.0, 2.0]


def test_no_warning_with_all_valid_keys():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning => test failure
        load_parameters(parms={"NUM_WIRES": 42, "T_PULSE": 1.0})


# ---------------------------------------------------------------------------
# Explicit filepath
# ---------------------------------------------------------------------------
def test_missing_filepath_raises(tmp_path):
    missing = tmp_path / "does_not_exist.ini"
    with pytest.raises(FileNotFoundError):
        load_parameters(filepath=missing)


def test_explicit_filepath_is_used(tmp_path):
    ini = tmp_path / "custom.ini"
    ini.write_text(
        "\n".join(
            [
                "[Network]",
                "num_wires = 999",
                "area = 1000.0",
                "length = 70.0",
                "proximity_threshold_ratio = 0.05",
                "",
                "[Electrical]",
                "g_leak = 1e-12",
                "v_input = 3.6",
                "v_ground = 0.0",
                "v_read = 0.05",
                "rho_plata = 0.0159",
                "diametro_nm = 115.0",
                "",
                "[Time]",
                "time_step_dt = 1e-3",
                "t_pulse = 10.0",
                "t_relax = 40.0",
                "",
                "[Memristor]",
                "g_on = 1e-3",
                "g_off = 1e-9",
                "v_threshold = 0.01",
                'evolver_model = "stochastic1"',
                "",
                "[Probabilities]",
                "p0_set = 0.4",
                "alpha_set = 1.0",
                "p_decay = 0.8",
                "beta_reset = 1.0",
                "seed = 42",
            ]
        ),
        encoding="utf-8",
    )
    result = load_parameters(filepath=ini)
    assert result["parameters"]["NUM_WIRES"] == 999


def test_evolver_section_in_ini(tmp_path):
    from importlib.resources import files

    base = files("neuromorphic").joinpath("defaults.ini").read_text(encoding="utf-8")
    ini = tmp_path / "with_evolver.ini"
    ini.write_text(
        base + "\n[Evolver]\nnu_up = 1e3\nmax_jumps = 2\nlabel = demo\nrates = [1.0, 2.0]\n",
        encoding="utf-8",
    )
    p = load_parameters(filepath=ini)["parameters"]
    assert p["NU_UP"] == 1e3
    assert p["MAX_JUMPS"] == 2 and isinstance(p["MAX_JUMPS"], int)
    assert p["LABEL"] == "demo"
    assert p["RATES"] == [1.0, 2.0]
