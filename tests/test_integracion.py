"""End-to-end tests across the full simulation pipeline."""

import numpy as np

from neuromorphic import setup_simulation
from neuromorphic.visualizacion import plot_simulation_results


def test_full_pipeline_runs(_pulse_run):
    _, t, g, act = _pulse_run
    assert len(t) == len(g) == len(act)
    assert any(abs(gi) > 0 for gi in g)


def test_simulation_dict_has_expected_keys(sim_percolating):
    for key in ("parameters", "junctions", "graph", "terminals", "circuit"):
        assert key in sim_percolating


def test_plot_runs_without_error(_pulse_run, tmp_path, monkeypatch):
    """Smoke test for the plotting function (Agg backend)."""
    import matplotlib.pyplot as plt

    sim, t, g, _ = _pulse_run
    out = tmp_path / "fig.png"
    monkeypatch.setattr(plt, "show", lambda: None)
    plot_simulation_results(sim, t, g)
    plt.savefig(out)
    plt.close("all")
    assert out.exists()


def test_setup_is_deterministic_with_seed(_sim_percolating_base, percolating_params):
    """Reconstruir con la misma semilla produce un grafo idéntico al cacheado."""
    np.random.seed(42)
    sim_fresh = setup_simulation(parms=percolating_params)

    assert sim_fresh["graph"].number_of_nodes() == _sim_percolating_base["graph"].number_of_nodes()
    assert sim_fresh["graph"].number_of_edges() == _sim_percolating_base["graph"].number_of_edges()
