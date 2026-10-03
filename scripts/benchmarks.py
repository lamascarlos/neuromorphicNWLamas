#!/usr/bin/env python
"""Reproducible benchmark for the neuromorphic simulator.

Usage
-----
    python scripts/bench.py
    python scripts/bench.py --n-wires 2000 --steps 5000
    python scripts/bench.py --seed 7 --repeat 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.sparse.linalg import spsolve

from neuromorphic import setup_simulation
from neuromorphic.fisica import (
    build_admittance_matrix,
    calculate_input_current,
    update_stochastic_conductance,
)


def timeit(fn, repeat: int = 1) -> tuple[float, object]:
    """Run fn repeat times; return (avg_seconds, last_result)."""
    t0 = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = fn()
    dt = (time.perf_counter() - t0) / repeat
    return dt, result


def format_row(label: str, seconds: float, width: int = 38) -> str:
    return f"{label:<{width}s} {seconds * 1000:>10.2f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-wires", type=int, default=1500, help="Number of nanowires (default: 1500)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Target steps for the total estimate (default: 5000)",
    )
    parser.add_argument(
        "--repeat", type=int, default=10, help="Repetitions per per-step measurement (default: 10)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    parms = {
        "NUM_WIRES": args.n_wires,
        "T_PULSE": 1.0,
        "T_RELAX": 1.0,
        "TIME_STEP_DT": 1e-2,
    }

    # ------------------------------------------------------------------
    # Setup (one-time)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    sim = setup_simulation(parms=parms)
    setup_time = time.perf_counter() - t0

    n_nodes = sim["graph"].number_of_nodes()
    n_edges = sim["graph"].number_of_edges()
    n_mem = sum(1 for _, _, d in sim["graph"].edges(data=True) if d.get("is_memristor", False))

    print()
    print(f"Network: {n_nodes} nodes, {n_edges} edges, {n_mem} memristors")
    print(f"Seed={args.seed}  N={args.n_wires}  repeat={args.repeat}")
    print("-" * 52)
    print(format_row("setup_simulation (one-time)", setup_time))
    print()

    # ------------------------------------------------------------------
    # Per-step components
    # ------------------------------------------------------------------
    Y, I_vec, _ = build_admittance_matrix(sim)
    V = spsolve(Y, I_vec)

    t_build, _ = timeit(lambda: build_admittance_matrix(sim), repeat=args.repeat)
    t_solve, _ = timeit(lambda: spsolve(Y, I_vec), repeat=args.repeat)
    t_update, _ = timeit(lambda: update_stochastic_conductance(sim, V), repeat=args.repeat)
    t_current, _ = timeit(lambda: calculate_input_current(sim, V), repeat=args.repeat)

    per_step_total = t_build + t_solve + t_update + t_current
    print(format_row("build_admittance_matrix", t_build))
    print(format_row("spsolve", t_solve))
    print(format_row("update_stochastic_conductance", t_update))
    print(format_row("calculate_input_current", t_current))
    print("-" * 52)
    print(format_row("per-step total", per_step_total))
    print()
    print(
        f"Estimated full run ({args.steps} steps): {setup_time + per_step_total * args.steps:.1f} s"
    )
    print()


if __name__ == "__main__":
    main()
