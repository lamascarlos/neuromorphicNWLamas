# admitancia.py
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix


# ==============================================================================
# PRECÓMPUTO DE ESTRUCTURAS (una vez por simulación)
# ==============================================================================
def _precompute_circuit_arrays(simulation: dict[str, Any]) -> None:
    """Cachea arrays con la estructura del grafo para evitar loops Python.

    Calcula y guarda en ``simulation["circuit"]``:

    - ``node_to_index``: mapeo nodo -> índice.
    - ``edge_u_idx``, ``edge_v_idx``: índices de los extremos por arista.
    - ``edge_is_memristor``: máscara booleana.
    - ``edge_fixed_g``: conductancia fija por arista (0 en memristores).
    - ``mem_edge_idx``: índices (en el array de aristas) de los memristores.
    - ``memristor_g``: conductancias actuales de los memristores.
    - ``input_idx``, ``output_idx``: índices de los electrodos.
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    circuit = simulation["circuit"]

    nodes = list(G.nodes)
    node_to_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    edges = list(G.edges(data=True))
    E = len(edges)

    u_idx = np.empty(E, dtype=np.int64)
    v_idx = np.empty(E, dtype=np.int64)
    is_mem = np.empty(E, dtype=bool)
    fixed_g = np.empty(E, dtype=np.float64)

    mem_count = 0
    for i, (u, v, data) in enumerate(edges):
        u_idx[i] = node_to_index[u]
        v_idx[i] = node_to_index[v]
        mem = data.get("is_memristor", False)
        is_mem[i] = mem
        if mem:
            fixed_g[i] = 0.0
            data["_mem_idx"] = mem_count  # índice del memristor en memristor_g
            mem_count += 1
        else:
            fixed_g[i] = 1.0 / (data["weight"] * p["R_WIRE_PER_LENGTH"] + 1e-12)

    mem_edge_idx = np.flatnonzero(is_mem)
    mem_g = np.empty(mem_count, dtype=np.float64)
    for j, edge_pos in enumerate(mem_edge_idx):
        mem_g[j] = edges[edge_pos][2]["conductance"]

    input_idx = np.array(
        [node_to_index[n] for n in simulation["terminals"]["input_nodes"]],
        dtype=np.int64,
    )
    output_idx = np.array(
        [node_to_index[n] for n in simulation["terminals"]["output_nodes"]],
        dtype=np.int64,
    )

    circuit["node_to_index"] = node_to_index
    circuit["N"] = N
    circuit["edge_u_idx"] = u_idx
    circuit["edge_v_idx"] = v_idx
    circuit["edge_is_memristor"] = is_mem
    circuit["edge_fixed_g"] = fixed_g
    circuit["mem_edge_idx"] = mem_edge_idx
    circuit["memristor_g"] = mem_g
    circuit["mem_u_idx"] = u_idx[mem_edge_idx]
    circuit["mem_v_idx"] = v_idx[mem_edge_idx]
    circuit["mem_edge_keys"] = [edges[i][:2] for i in mem_edge_idx]
    circuit["input_idx"] = input_idx
    circuit["output_idx"] = output_idx


# ==============================================================================
# CONSTRUCCIÓN DE LA MATRIZ DE ADMITANCIA (SPARSE)
# ==============================================================================
def build_admittance_matrix(
    simulation: dict[str, Any],
    v_input: float | None = None,
):
    """Ensambla la matriz de admitancia y el vector de corrientes.

    Construye el sistema :math:`Y \\cdot V = I` en formato COO → CSR
    usando arrays de NumPy vectorizados. La estructura del grafo se
    precalcula una sola vez y se reutiliza en cada paso.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"`` y ``"terminals"``.
    v_input : float, optional
        Voltaje de entrada a fijar en esta llamada. Si es ``None``
        (default), se usa ``parameters["V_INPUT"]``.

    Returns
    -------
    Y : scipy.sparse.csr_matrix
        Matriz de admitancia dispersa de shape ``(N, N)``.
    I_vec : numpy.ndarray
        Vector de corrientes externas de shape ``(N,)``.
    node_to_index : dict
        Mapeo ``node_id -> row_index``.

    Notes
    -----
    También deja el resultado disponible en ``simulation["circuit"]``
    como ``{"Y", "I", "node_to_index", ...}``.

    El código asume que la red percola y que ``prune_dead_components``
    fue ejecutado. Con nodos flotantes y ``G_LEAK`` pequeño, la matriz
    puede quedar numéricamente singular.
    """
    p = simulation["parameters"]
    if "circuit" not in simulation:
        simulation["circuit"] = {}
    circuit = simulation["circuit"]

    if "edge_u_idx" not in circuit:
        _precompute_circuit_arrays(simulation)

    if v_input is None:
        v_input = p["V_INPUT"]
    v_ground = p["V_GROUND"]
    g_leak = p["G_LEAK"]

    u_idx = circuit["edge_u_idx"]
    v_idx = circuit["edge_v_idx"]
    fixed_g = circuit["edge_fixed_g"]
    mem_edge_idx = circuit["mem_edge_idx"]
    mem_g = circuit["memristor_g"]
    N = circuit["N"]
    input_idx = circuit["input_idx"]
    output_idx = circuit["output_idx"]

    # --- Conductancia por arista (segmentos fijos + memristores actuales) ---
    edge_g = fixed_g.copy()
    if len(mem_edge_idx) > 0:
        edge_g[mem_edge_idx] = mem_g

    # --- Construir triplets COO: 4 por arista ---
    E = len(u_idx)
    rows = np.empty(4 * E, dtype=np.int64)
    cols = np.empty(4 * E, dtype=np.int64)
    vals = np.empty(4 * E, dtype=np.float64)

    rows[0::4] = u_idx
    cols[0::4] = u_idx
    vals[0::4] = edge_g

    rows[1::4] = v_idx
    cols[1::4] = v_idx
    vals[1::4] = edge_g

    rows[2::4] = u_idx
    cols[2::4] = v_idx
    vals[2::4] = -edge_g

    rows[3::4] = v_idx
    cols[3::4] = u_idx
    vals[3::4] = -edge_g

    # --- G_LEAK en diagonales de nodos internos ---
    electrode_mask = np.zeros(N, dtype=bool)
    electrode_mask[input_idx] = True
    electrode_mask[output_idx] = True

    leak_rows = np.arange(N, dtype=np.int64)
    leak_vals = np.where(electrode_mask, 0.0, g_leak)

    all_rows = np.concatenate([rows, leak_rows])
    all_cols = np.concatenate([cols, leak_rows])
    all_vals = np.concatenate([vals, leak_vals])

    Y = coo_matrix((all_vals, (all_rows, all_cols)), shape=(N, N)).tocsr()

    # --- Condiciones de contorno de Dirichlet (fila identidad) ---
    I_vec = np.zeros(N)

    for idx in input_idx:
        s, e = Y.indptr[idx], Y.indptr[idx + 1]
        cols_in_row = Y.indices[s:e]
        Y.data[s:e] = np.where(cols_in_row == idx, 1.0, 0.0)
        I_vec[idx] = v_input

    for idx in output_idx:
        s, e = Y.indptr[idx], Y.indptr[idx + 1]
        cols_in_row = Y.indices[s:e]
        Y.data[s:e] = np.where(cols_in_row == idx, 1.0, 0.0)
        I_vec[idx] = v_ground

    Y.eliminate_zeros()

    circuit["Y"] = Y
    circuit["I"] = I_vec
    return Y, I_vec, circuit["node_to_index"]
