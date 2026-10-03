# simulador.py
from collections.abc import Callable
from typing import Any

from scipy.sparse.linalg import spsolve

from . import fisica as fis
from .grafo import check_percolation

SimulationCallbackFunction = Callable[[float, dict[str, Any], dict[str, Any]], None]


# ==============================================================================
# SIMULACIÓN DE PULSOS DINÁMICOS
# ==============================================================================
def run_simulation_dynamic_pulse(
    simulation: dict[str, Any], callback: SimulationCallbackFunction | None = None
):
    """Ejecuta el experimento de pulso y relajación (Fig. 2b).

    Somete la red a un voltaje alto ``V_INPUT`` durante ``T_PULSE``
    segundos (fase de facilitación) y luego a un voltaje bajo ``V_READ``
    durante ``T_RELAX`` segundos (fase de relajación). En cada paso
    temporal se resuelve el circuito, se registra la conductancia total
    y se actualiza estocásticamente el estado de los memristores.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"``, ``"terminals"`` y ``"circuit"``.
    callback : callable, optional
        Función ``callback(t, simulation, step_data)`` que se llama en cada
        paso, después de actualizar la conductancia. Recibe el tiempo ``t``
        del paso, el diccionario de simulación (ya con el estado
        actualizado) y un diccionario ``step_data`` con los datos del
        circuito resuelto en ese paso, es decir, *antes* de la
        actualización:

        - ``"Y"``: matriz de admitancia;
        - ``"I_vec"``: vector de corrientes inyectadas;
        - ``"V_vec"``: voltajes nodales;
        - ``"V_input"``: voltaje aplicado en los electrodos de entrada (V);
        - ``"G_total"``: conductancia equivalente de la red (mS), el mismo
          valor que se agrega al historial.

        Permite guardar información calculada en cada paso, hacer
        verificaciones al vuelo o interrumpir la simulación lanzando una
        excepción, que se propaga a quien llamó a esta función (el
        historial acumulado hasta ese momento se pierde).

    Returns
    -------
    history_time : list of float
        Instantes de tiempo registrados (s).
    history_G_total : list of float
        Conductancia equivalente total de la red en cada paso (mS).
    history_active : list of int
        Cantidad de memristores en estado ``ON`` en cada paso.

    Raises
    ------
    RuntimeError
        Si la red no tiene electrodos o si no percola topológicamente.

    Notes
    -----
    Al comenzar se llama a :func:`~neuromorphic.fisica.initialize_evolver`,
    que reinicia el estado de los memristores según el modelo elegido en
    ``EVOLVER``. Correr dos veces la misma simulación parte del mismo
    estado inicial. Al terminar, el grafo, ``circuit["memristor_g"]`` y
    ``simulation["evolver_state"]`` quedan con el estado final.
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    terminals = simulation["terminals"]
    input_nodes = terminals["input_nodes"]
    output_nodes = terminals["output_nodes"]

    if len(input_nodes) == 0 or len(output_nodes) == 0:
        raise RuntimeError(
            "La red no tiene electrodos conectados: "
            f"input_nodes={len(input_nodes)}, output_nodes={len(output_nodes)}"
        )

    # --- Diagnóstico de conectividad ---
    if not check_percolation(simulation):
        # Estimar densidad vs umbral crítico
        L = p["LENGTH"]
        A = p["AREA"]
        N = p["NUM_WIRES"]
        N_c = 5.7 * A / (L**2)
        raise RuntimeError(
            f"La red NO percola: no hay camino topológico entre electrodos.\n"
            f"  Densidad actual : N = {N} (N·L²/A = {N * L**2 / A:.2f})\n"
            f"  Umbral crítico  : N_c ≈ {N_c:.0f} (N·L²/A ≈ 5.7)\n"
            f"  Sugerencia      : aumentar NUM_WIRES > {int(N_c * 1.3)} "
            f"o agrandar LENGTH a > {int((5.7 * A / N) ** 0.5 * 1.3)} µm."
        )

    N_orig_junctions = len(simulation["junctions"]["junctions"])
    print(
        f"--- Simulación de Red de Nanohilos N = {p['NUM_WIRES']} (Pulso y Relajación - Fig 2b) ---"
    )
    print(
        f"Memristores: {N_orig_junctions} | Nodos de grafo: {G.number_of_nodes()} | "
        f"Inputs: {len(input_nodes)} | Outputs: {len(output_nodes)}"
    )

    # Estado inicial de los memristores según el modelo de evolución.
    # Reinicia array, grafo, máscara de activos y evolver_state.
    update_conductance = fis.initialize_evolver(simulation).update
    circuit = simulation["circuit"]

    history_time = []
    history_G_total = []
    history_active = []
    current_time = 0.0

    T_PULSE = p["T_PULSE"]
    T_RELAX = p["T_RELAX"]
    V_PULSE = p["V_INPUT"]
    V_READ = p["V_READ"]
    dt = p["TIME_STEP_DT"]
    total_steps = int((T_PULSE + T_RELAX) / dt)

    print(f"Iniciando: {T_PULSE}s pulso ({V_PULSE}V) + {T_RELAX}s relajación ({V_READ}V)")

    for step in range(total_steps):
        # Voltaje dinámico según la fase (sin mutar p['V_INPUT'])
        v_now = V_PULSE if current_time <= T_PULSE else V_READ

        # 1) Resolver el circuito con el voltaje actual
        Y, I_vec, _ = fis.build_admittance_matrix(simulation, v_input=v_now)
        try:
            V_vec = spsolve(Y, I_vec)
        except Exception as e:
            print(f"Error en spsolve en t={current_time:.3f} s: {e}")
            break

        # 2) Conductancia equivalente total
        I_in = fis.calculate_input_current(simulation, V_vec)
        G_total = 1000.0 * (I_in / v_now) if v_now != 0 else 0.0

        # 3) Contar memristores activos (máscara mantenida por el evolver)
        count_ON = int(circuit["memristor_active"].sum())

        history_time.append(current_time)
        history_G_total.append(G_total)
        history_active.append(count_ON)

        # 4) Actualización estocástica (muta G in-place)
        update_conductance(simulation, V_vec)

        # 5) Hook del usuario
        if callback is not None:
            step_data = {
                "Y": Y,
                "I_vec": I_vec,
                "V_vec": V_vec,
                "V_input": v_now,
                "G_total": G_total,
            }
            callback(current_time, simulation, step_data)

        current_time += dt

        if step % max(1, total_steps // 10) == 0:
            mode = "PULSO" if current_time <= T_PULSE else "RELAX"
            print(f"[{mode}] T={current_time:.1f}s | G={G_total:.4e} mS | ON={count_ON}")

    return history_time, history_G_total, history_active
