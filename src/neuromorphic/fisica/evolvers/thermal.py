"""Modelo ``thermal``: filamento con variable de estado continua ``lam``.

Reformulación de ``junction_update.py``. Por juntura:

- ``lam = 0``: sin filamento (gap de PVP completo, régimen túnel);
- ``lam = 1``: el filamento cierra el gap (``G = G0``);
- ``1 < lam <= THERMAL_LAM_MAX``: el filamento se engrosa (hasta ``N_MAX·G0``).

En cada paso, con la temperatura local por efecto Joule
``T = T0 + Rth·G·V²``:

1. crecimiento por migración iónica a campo alto (Mott-Gurney), con el
   campo creciente al cerrarse el gap;
2. disolución activada térmicamente, más lenta cuanto mayor es ``lam``;
3. ruptura si ``T > THERMAL_T_RUPT``.

El modelo es determinista. ``lam`` vive en ``evolver_state["lam"]`` y los
diagnósticos del último paso (``T``, ``saturated``, ``ruptured``) también.
Todos los parámetros pueden ser arrays por juntura.
"""

import warnings
from typing import Any

import numpy as np

from ._junction import G0, JUNCTION_GEOMETRY_DEFAULTS, KB, junction_drive, set_conductances
from .base import register_memristor_evol_model

THERMAL_PARAMETERS: dict[str, Any] = {
    **JUNCTION_GEOMETRY_DEFAULTS,
    "THERMAL_NU": 1e13,  # 1/s, frecuencia de intento
    "THERMAL_EA": 0.6,  # eV, barrera de migración de Ag+ en PVP
    "THERMAL_ED": 0.4,  # eV, barrera de la disolución
    "THERMAL_TAU0": 1e-6,  # s, prefactor del tiempo de disolución
    "THERMAL_BETA": 3.0,  # estabilización de filamentos gruesos
    "THERMAL_LAM_MAX": 3.0,
    "THERMAL_RTH": 1e7,  # K/W, resistencia térmica efectiva
    "THERMAL_T0": 300.0,  # K
    "THERMAL_T_RUPT": 900.0,  # K, umbral de ruptura térmica
    "THERMAL_DLAM_MAX": 0.05,  # máximo cambio de lam por paso (control numérico)
}


class StepSaturationWarning(RuntimeWarning):
    """El paso temporal es demasiado grande para la dinámica de ``lam``."""


def thermal_conductance(lam: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Conductancia de la juntura: túnel para ``lam < 1``, metálica para ``lam >= 1``."""
    lam = np.asarray(lam, dtype=float)
    gap = p["GAP_D"] * np.clip(1.0 - lam, 0.0, 1.0)
    g_tun = G0 * np.exp(-2.0 * p["KAPPA"] * gap)
    frac = np.clip((lam - 1.0) / (p["THERMAL_LAM_MAX"] - 1.0), 0.0, 1.0)
    g_met = G0 * (1.0 + (p["N_MAX"] - 1.0) * frac)
    return np.where(lam < 1.0, g_tun, g_met)


def init_thermal(simulation: dict[str, Any]) -> None:
    p = simulation["parameters"]
    n = simulation["circuit"]["memristor_g"].size
    state = simulation["evolver_state"]
    state["lam"] = np.zeros(n)
    state["T"] = np.broadcast_to(np.asarray(p["THERMAL_T0"], dtype=float), (n,)).copy()
    state["saturated"] = np.zeros(n, dtype=bool)
    state["ruptured"] = np.zeros(n, dtype=bool)
    state["saturation_warned"] = False
    set_conductances(simulation, thermal_conductance(state["lam"], p))


@register_memristor_evol_model("thermal", init=init_thermal, parameters=THERMAL_PARAMETERS)
def thermal_updater(simulation: dict[str, Any], V_solved):
    """Avanza ``lam`` un paso ``dt`` con la caída de tensión de cada juntura.

    Emite :class:`StepSaturationWarning` (una vez por corrida) si en alguna
    juntura el paso se recortó a ``THERMAL_DLAM_MAX``: es señal de que
    conviene reducir ``TIME_STEP_DT``.
    """
    p = simulation["parameters"]
    state = simulation["evolver_state"]
    dt = p["TIME_STEP_DT"]
    lam = state["lam"]
    d = p["GAP_D"]
    a = p["HOP_A"]

    v_mem, i_mem = junction_drive(simulation, V_solved)

    # Temperatura local por efecto Joule (explícita: usa G del paso actual)
    T = p["THERMAL_T0"] + p["THERMAL_RTH"] * i_mem * v_mem
    kT = KB * T

    # 1) Crecimiento: migración iónica a campo alto (Mott-Gurney)
    gap = np.maximum(d * (1.0 - np.minimum(lam, 1.0)), a)
    x = np.minimum(a * v_mem / gap / (2.0 * kT), 50.0)
    v_ion = a * p["THERMAL_NU"] * np.exp(-p["THERMAL_EA"] / kT) * np.sinh(x)
    dlam_raw = v_ion / d * dt
    saturated = dlam_raw > p["THERMAL_DLAM_MAX"]
    dlam = np.minimum(dlam_raw, p["THERMAL_DLAM_MAX"])

    # 2) Disolución espontánea, más lenta para filamentos gruesos
    tau = p["THERMAL_TAU0"] * np.exp(p["THERMAL_ED"] / kT) * np.exp(p["THERMAL_BETA"] * lam)
    lam_new = (lam + dlam) * np.exp(-dt / tau)

    # 3) Ruptura térmica por corriente excesiva
    ruptured = p["THERMAL_T_RUPT"] < T
    lam_new = np.where(ruptured, 0.0, lam_new)
    lam_new = np.clip(lam_new, 0.0, p["THERMAL_LAM_MAX"])

    state["lam"] = lam_new
    state["T"] = T
    state["saturated"] = saturated
    state["ruptured"] = ruptured
    if saturated.any() and not state["saturation_warned"]:
        state["saturation_warned"] = True
        warnings.warn(
            f"{int(saturated.sum())} junturas recortaron dlam a THERMAL_DLAM_MAX "
            f"con TIME_STEP_DT={dt}; considerá reducir el paso.",
            StepSaturationWarning,
            stacklevel=2,
        )

    set_conductances(simulation, thermal_conductance(lam_new, p))
    return simulation["graph"]
