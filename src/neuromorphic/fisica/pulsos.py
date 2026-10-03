# pulsos.py

import numpy as np


# ==============================================================================
# SEÑALES DE ENTRADA EXTRA
# ==============================================================================
def get_v_ramp(t: float, total_time: float, amplitude: float) -> float:
    """Señal senoidal para barridos de histéresis.

    Parameters
    ----------
    t : float
        Tiempo actual.
    total_time : float
        Período completo de la señal.
    amplitude : float
        Amplitud pico (V).

    Returns
    -------
    float
        Voltaje instantáneo :math:`V_0 \\sin(2 \\pi t / T)`.
    """
    return amplitude * np.sin(2 * np.pi * t / total_time)
