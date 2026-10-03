# geometria.py
from typing import Any

import numpy as np


# ==============================================================================
# FUNCIONES DE GEOMETRÍA Y RED
# ==============================================================================
def generate_and_find_junctions(simulation: dict[str, Any]) -> None:
    """Genera la disposición espacial de los nanohilos y encuentra sus cruces.

    Vectorizado con NumPy: la detección de intersecciones pasa de
    :math:`O(N^2)` en Python puro a broadcasting vectorizado.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"`` con las
        claves ``NUM_WIRES``, ``AREA`` y ``LENGTH``.

    Returns
    -------
    None
        Modifica ``simulation`` in-place agregando la clave ``"junctions"``.

    Notes
    -----
    La secuencia de llamadas RNG se preserva idénticamente respecto de la
    versión original, así que con la misma semilla se obtienen los mismos
    hilos y las mismas junturas.

    La construcción de la matriz ``(N, N)`` de distancias consume
    :math:`O(N^2)` memoria temporal (aprox. 200 MB para ``N=2000``).
    Para ``N > 5000`` conviene chunkear.
    """
    p = simulation["parameters"]
    num_wires = p["NUM_WIRES"]
    wire_length = p["LENGTH"]
    area_size = p["AREA"]

    # --- RNG: orden y argumentos idénticos a la versión original ---
    xc = np.random.uniform(0, area_size, num_wires)
    yc = np.random.uniform(0, area_size, num_wires)
    theta = np.random.uniform(0, np.pi, num_wires)

    x_off = (wire_length / 2) * np.cos(theta)
    y_off = (wire_length / 2) * np.sin(theta)

    # Endpoints como arrays (N, 2)
    p1_arr = np.stack([xc - x_off, yc - y_off], axis=1)
    p2_arr = np.stack([xc + x_off, yc + y_off], axis=1)
    d_arr = p2_arr - p1_arr

    wires = [{"id": i, "p1": p1_arr[i].copy(), "p2": p2_arr[i].copy()} for i in range(num_wires)]

    # --- Detección vectorizada de intersecciones ---
    p1_x, p1_y = p1_arr[:, 0], p1_arr[:, 1]
    d_x, d_y = d_arr[:, 0], d_arr[:, 1]

    # denom[i, j] = d_x[i] * d_y[j] - d_y[i] * d_x[j]
    denom = d_x[:, None] * d_y[None, :] - d_y[:, None] * d_x[None, :]

    # Diferencias de punto inicial: p1[j] - p1[i]
    dx_diff = p1_x[None, :] - p1_x[:, None]
    dy_diff = p1_y[None, :] - p1_y[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        t_num = dx_diff * d_y[None, :] - dy_diff * d_x[None, :]
        u_num = dx_diff * d_y[:, None] - dy_diff * d_x[:, None]
        t = t_num / denom
        u = u_num / denom

    valid = (denom != 0.0) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    # Quedarnos solo con i < j.
    valid = np.triu(valid, k=1)

    # np.nonzero devuelve en orden row-major (i, j), idéntico al
    # orden de descubrimiento del doble loop original.
    i_idx, j_idx = np.nonzero(valid)
    n_junctions = len(i_idx)

    # Coordenadas de intersección: p1[i] + t * d[i]
    ix = p1_x[i_idx] + t[i_idx, j_idx] * d_x[i_idx]
    iy = p1_y[i_idx] + t[i_idx, j_idx] * d_y[i_idx]

    junctions: list[dict[str, Any]] = []
    wire_to_junctions: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_wires)}

    for k in range(n_junctions):
        i = int(i_idx[k])
        j = int(j_idx[k])
        j_data = {
            "id": k,
            "pos": np.array([ix[k], iy[k]]),
            "wires": (i, j),
        }
        junctions.append(j_data)
        wire_to_junctions[i].append(j_data)
        wire_to_junctions[j].append(j_data)

    simulation["junctions"] = {
        "wires": wires,
        "junctions": junctions,
        "wire_to_junctions": wire_to_junctions,
    }
