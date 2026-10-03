"""Kernel Numba del modelo ``ladder_numba``. Importa numba al cargarse."""

import numba
import numpy as np


@numba.njit(cache=True)
def seed(s):
    """Siembra el generador de Numba (independiente del de NumPy)."""
    np.random.seed(s)


@numba.njit(cache=True)
def ladder_step(G, V, I_mem, dt, ladder, gfac, edges, rates, out):  # pragma: no cover
    """Actualiza ``G -> out`` (``out`` puede ser ``G``).

    ``rates`` tiene shape ``(9, n)``: filas ``nu_up, V_s, nu_dn, beta, P_T,
    nu_rp, P_c, m, max_jumps``, una columna por juntura.

    El bucle es secuencial a propósito: con ``prange`` cada hilo usa su
    propio generador y la corrida deja de ser reproducible con la semilla.
    """
    kmax = ladder.size - 1
    for j in range(G.size):
        nu_up = rates[0, j]
        V_s = rates[1, j]
        nu_dn = rates[2, j]
        beta = rates[3, j]
        P_T = rates[4, j]
        nu_rp = rates[5, j]
        P_c = rates[6, j]
        m = rates[7, j]
        max_jumps = int(rates[8, j])

        k = np.searchsorted(edges, G[j])
        P = abs(I_mem[j] * V[j])

        # Ruptura (solo se sortea si la probabilidad no es despreciable)
        p_rp = nu_rp * (P / P_c) ** m * dt
        if p_rp > 1e-12 and np.random.random() < -np.expm1(-p_rp):
            out[j] = ladder[0]
            continue

        # Crecimiento (campo en el gap) y disolución (asistida por Joule)
        x = min(abs(V[j]) / (V_s * gfac[k]), 50.0)
        a_up = nu_up * np.sinh(x) * dt
        a_dn = nu_dn * np.exp(-beta * k + min(P / P_T, 50.0)) * dt
        a_tot = a_up + a_dn
        if a_tot < 1e-12:
            out[j] = ladder[k]
            continue

        # N ~ Poisson(a_tot) truncado en max_jumps, por transformada inversa;
        # cada evento es "arriba" con probabilidad a_up / a_tot.
        u = np.random.random()
        prob = np.exp(-a_tot)
        cdf = prob
        n = 0
        while u > cdf and n < max_jumps:
            n += 1
            prob *= a_tot / n
            cdf += prob
        dk = 0
        q_up = a_up / a_tot
        for _ in range(n):
            dk += 1 if np.random.random() < q_up else -1
        out[j] = ladder[min(max(k + dk, 0), kmax)]
