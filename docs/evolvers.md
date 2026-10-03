# Modelos de evolución de memristores

## Resumen

Al final de cada paso temporal del motor (`run_simulation_dynamic_pulse`),
el estado de cada memristor se actualiza invocando una función
*evolver*. Qué modelo se usa se selecciona con la clave `evolver_model`
del `.ini` —o su equivalente `EVOLVER` en los overrides— y la resolución
se hace contra un registro global:

~~~python
update_conductance = fis.initialize_evolver(simulation).update
~~~

El paquete expone un mecanismo de registro para agregar modelos propios
sin modificar el código fuente.

## Selección del modelo

### Desde un `.ini`

~~~ini
[Memristor]
g_on = 1e-3
g_off = 1e-9
v_threshold = 0.01
evolver_model = stochastic1
~~~

### Desde `parms=`

~~~python
sim = setup_simulation(parms={"EVOLVER": "stochastic1"})
~~~

El valor debe coincidir con un modelo registrado en
`neuromorphic.fisica.EVOLVER_SPECS`. Si no existe, `setup_simulation`
falla con `KeyError` y la lista de modelos disponibles.

## Componentes de un modelo

Un modelo se registra con

~~~python
@register_memristor_evol_model(name, init=None, parameters=None)
def update(simulation, V_solved): ...
~~~

y queda descrito por un `EvolverSpec(update, init, parameters)`:

| Componente | Rol |
|---|---|
| `update(simulation, V_solved)` | Avanza un paso temporal. Obligatorio. |
| `init(simulation)` | Fija el estado inicial y precalcula lo que el modelo necesite. Opcional; por defecto, todos los memristores en `G_OFF`. |
| `parameters` | Dict `{CLAVE: valor_por_defecto}` con los parámetros propios del modelo. Opcional. |

### Ciclo de vida

`initialize_evolver(simulation)` prepara el modelo indicado en
`parameters["EVOLVER"]`:

1. completa en `simulation["parameters"]` las claves de `parameters` que
   falten, sin pisar valores ya definidos por el `.ini` o por `parms=`;
2. recrea `simulation["evolver_state"]` como un dict vacío;
3. llama a `init(simulation)`.

La llaman `setup_simulation`, al final, y `run_simulation_dynamic_pulse`,
al comienzo de cada corrida. Por eso correr dos veces la misma
simulación parte siempre del mismo estado inicial.

### Parámetros propios

Las claves declaradas en `parameters` se aceptan en `parms=` sin
advertencias y conviven con el resto en `simulation["parameters"]`.
Pueden ser escalares o arrays con un valor por memristor (en el orden de
`memristor_g`). También pueden definirse en el `.ini`, en una sección
libre `[Evolver]`: cada clave se pasa a mayúsculas y su valor se
interpreta como literal de Python.

~~~ini
[Evolver]
nu_up = 1e3
max_jumps = 2
~~~

## Contrato de `update`

~~~python
def mi_evolver(
    simulation: dict[str, Any],
    V_solved: np.ndarray,
) -> "networkx.Graph": ...
~~~

Recibe el diccionario completo de simulación y el vector de voltajes
nodales recién resuelto (`Y · V = I`). Debe mantener coherentes tres
estructuras (`init` también debe dejarlas coherentes):

| Estructura | Descripción |
|---|---|
| `simulation["circuit"]["memristor_g"]` | Array `float64` con la conductancia actual de cada memristor. Lo consume `build_admittance_matrix`. Se modifica in-place. |
| `simulation["graph"].edges[u, v]["conductance"]` | Valor espejo en las aristas del grafo. Lo usan el cálculo de corrientes, diagnósticos y visualización. |
| `simulation["circuit"]["memristor_active"]` | Máscara booleana de memristores "activos". El motor reporta su suma en cada paso. Para modelos binarios es `memristor_g == G_ON`; un modelo de conductancia continua define su propio criterio. |

El estado interno del modelo (variables por juntura, tablas
precalculadas, generadores aleatorios, diagnósticos) va en
`simulation["evolver_state"]`.

También están disponibles:

- `simulation["parameters"]` — parámetros crudos, derivados y propios del modelo.
- `simulation["circuit"]["mem_edge_keys"]` — lista de tuplas `(u, v)` en
  el mismo orden que `memristor_g`.
- `simulation["circuit"]["mem_u_idx"]`, `["mem_v_idx"]` — índices en
  `V_solved` de los extremos de cada memristor.
- `V_solved` — voltajes nodales del paso actual.

El valor de retorno se ignora; se recomienda devolver el grafo por
convención.

> **RNG**: `setup_simulation` siembra el generador global con
> `np.random.seed(RNG_SEED)`, y el evolver incluido (`stochastic1`)
> consume `numpy.random` directamente. Con la misma semilla y el mismo
> evolver la corrida es reproducible. Dos evolvers distintos con la misma
> semilla no producirán necesariamente las mismas conmutaciones, porque
> consumen el generador en distinto orden. Por ahora el paquete no admite
> inyectar un `np.random.Generator` global; si tu modelo lo necesita,
> crealo en `init` (por ejemplo con `np.random.default_rng(p["RNG_SEED"])`)
> y guardalo en `simulation["evolver_state"]`.

## Registrar un modelo propio

El ejemplo siguiente usa los tres componentes: un parámetro propio
(`P_FLIP`), un `init` que crea un generador aleatorio en
`evolver_state`, y un `update` que mantiene coherentes array, grafo y
máscara.

~~~python
import numpy as np
from neuromorphic.fisica import register_memristor_evol_model


def init_flip(simulation):
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]
    circuit["memristor_g"][:] = p["G_OFF"]
    for u, v in circuit["mem_edge_keys"]:
        G.edges[u, v]["conductance"] = p["G_OFF"]
    circuit["memristor_active"] = np.zeros(len(circuit["memristor_g"]), dtype=bool)
    simulation["evolver_state"]["rng"] = np.random.default_rng(p["RNG_SEED"])


@register_memristor_evol_model("flip", init=init_flip, parameters={"P_FLIP": 0.01})
def flip(simulation, V_solved):
    """Ejemplo trivial: cada memristor invierte su estado con prob. P_FLIP."""
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]
    rng = simulation["evolver_state"]["rng"]

    mem_g = circuit["memristor_g"]
    flip = rng.random(mem_g.size) < p["P_FLIP"]
    mem_g[flip] = np.where(mem_g[flip] == p["G_ON"], p["G_OFF"], p["G_ON"])
    for j in np.flatnonzero(flip):
        u, v = circuit["mem_edge_keys"][j]
        G.edges[u, v]["conductance"] = mem_g[j]
    circuit["memristor_active"] = mem_g == p["G_ON"]
    return G
~~~

Para el caso común de fijar todos los memristores en un mismo valor
existe el helper `set_all_memristors(simulation, g_value)`, que actualiza
array, grafo y máscara.

Una vez importado el módulo que contiene el decorador, el modelo queda
disponible para cualquier simulación del proceso:

~~~python
from neuromorphic import setup_simulation, run_simulation_dynamic_pulse

sim = setup_simulation(parms={"EVOLVER": "flip", "P_FLIP": 0.05})
t, G_total, activos = run_simulation_dynamic_pulse(sim)
~~~

El registro es **global al proceso**: importá el módulo del evolver una
sola vez (por ejemplo en tu `__init__.py` o al inicio del notebook) antes
de llamar a `setup_simulation`, para que sus parámetros se reconozcan
como declarados. Registrar un nombre que ya
existe reemplaza la función anterior sin emitir advertencia.

### Agregar un modelo al paquete

Para que un modelo forme parte del paquete, alcanza con crear un módulo
público (sin `_` inicial) en `src/neuromorphic/fisica/evolvers/` que lo
registre con el decorador. No hace falta tocar ningún `__init__.py`: el
paquete importa automáticamente todos sus módulos públicos.

Si el módulo necesita una dependencia opcional, no la importes en el
nivel superior del módulo público, porque se cargaría al importar
`neuromorphic`. Ponela en un módulo privado (por ejemplo `_mi_kernel.py`)
e importalo dentro del `init` del modelo, como hace `ladder_numba`.

## Modelos incluidos

| Nombre | Descripción |
|---|---|
| `stochastic1` | SET por sobretensión con P = P0·exp[α(V−V_th)] y RESET con P = P_decay·exp(−V/V_th). Las probabilidades son por paso y no escalan con `TIME_STEP_DT`. Modelo por defecto. |
| `stochastic2` | Modelo de Lamas et al. (2026): P_set = Δt·P0·max(0, 1 − exp[−α(V−V_th)]) y P_reset = min(1, Δt·P_decay·I²), con I = G_ON·V. `P0_SET` está en s⁻¹ y `P_DECAY` en A⁻²·s⁻¹. |

| `ladder` | Junturas Ag/PVP/Ag como proceso de nacimiento-muerte sobre una escalera discreta de conductancias (niveles túnel y metálicos `n·G0`). Crecimiento por campo, disolución asistida por Joule y ruptura por potencia. Estocástico, con generador propio sembrado con `RNG_SEED`. |
| `ladder_numba` | El mismo proceso con un kernel Numba. Requiere el extra `numba`. |
| `thermal` | Junturas Ag/PVP/Ag con variable de estado continua `lam` (0: gap abierto, 1: filamento cerrado, >1: filamento que se engrosa). Crecimiento Mott-Gurney, disolución térmica y ruptura por temperatura Joule. Determinista. |

> Los valores de `defaults.ini` están calibrados para `stochastic1`. Con
> `stochastic2` las mismas cifras tienen otras unidades: por ejemplo,
> con `G_ON = 1e-3` S y caídas de ~1 V, `P_DECAY = 0.8` da una
> probabilidad de RESET del orden de 10⁻⁹ por paso.

## Modelos de junturas Ag/PVP/Ag

`ladder`, `ladder_numba` y `thermal` describen la juntura con conductancias
físicas: arrancan en el nivel túnel más bajo, `G0·exp(-2·KAPPA·GAP_D)`
(≈ 3.5 nS con los valores por defecto), y llegan hasta `N_MAX·G0`
(≈ 0.77 mS). No usan `G_ON`, `G_OFF` ni `V_THRESHOLD`. Una juntura se
considera activa (`memristor_active`) cuando el filamento cierra el gap,
es decir `G >= G0`. En todos, la corriente por la juntura es
`I = G·V` con la conductancia del paso actual.

### Parámetros compartidos

Escalares en todos los modelos de este grupo:

| Clave | Default | Descripción |
|---|---|---|
| `GAP_D` | 1e-9 m | Espesor efectivo de PVP en la juntura. |
| `HOP_A` | 2.5e-10 m | Distancia de salto iónico. |
| `KAPPA` | 5e9 1/m | Decaimiento túnel. |
| `N_MAX` | 10 | Conductancia máxima en unidades de `G0`. |

### `ladder` y `ladder_numba`

El estado es el nivel `k` de la escalera, que se recupera de `G`: no hay
variables internas por juntura. `evolver_state` guarda la escalera
(`ladder`, `gfac`, `edges`) y el generador (`rng` en `ladder`). Las tasas
pueden ser arrays con un valor por juntura.

| Clave | Default | Descripción |
|---|---|---|
| `LADDER_NU_UP`, `LADDER_V_S` | 1e3 1/s, 0.2 V | Crecimiento: `r_up = NU_UP·sinh(|V| / (V_S·gap_k/d))`. |
| `LADDER_NU_DN`, `LADDER_BETA`, `LADDER_P_T` | 1 1/s, 0.7, 1e-5 W | Disolución: `r_dn = NU_DN·exp(-BETA·k)·exp(P/P_T)`. |
| `LADDER_NU_RP`, `LADDER_P_C`, `LADDER_M` | 1e3 1/s, 6e-5 W, 6 | Ruptura a `k = 0`: `r_rp = NU_RP·(P/P_C)^M`. |
| `LADDER_MAX_JUMPS` | 2 | Máximo de saltos por paso (escalar). |

`ladder_numba` lee las tasas al inicializar, así que cambiarlas a mitad
de corrida no tiene efecto. Trunca en `LADDER_MAX_JUMPS` el total de
eventos, mientras que `ladder` trunca subidas y bajadas por separado;
ambos coinciden cuando `r·dt` es chico. El kernel es secuencial para que
la corrida sea reproducible con la semilla. En redes del tamaño habitual
el paso está dominado por la resolución del circuito, así que la versión
Numba no acelera la simulación completa; sirve como referencia para
ensambles grandes de junturas.

### `thermal`

`evolver_state` guarda `lam` y los diagnósticos del último paso: la
temperatura `T` y las máscaras `saturated` y `ruptured`. Todos los
parámetros pueden ser arrays por juntura.

| Clave | Default | Descripción |
|---|---|---|
| `THERMAL_NU`, `THERMAL_EA` | 1e13 1/s, 0.6 eV | Migración iónica (Mott-Gurney). |
| `THERMAL_ED`, `THERMAL_TAU0`, `THERMAL_BETA` | 0.4 eV, 1e-6 s, 3 | Disolución: `tau = TAU0·exp(ED/kT)·exp(BETA·lam)`. |
| `THERMAL_LAM_MAX` | 3 | Máximo engrosamiento del filamento. |
| `THERMAL_RTH`, `THERMAL_T0`, `THERMAL_T_RUPT` | 1e7 K/W, 300 K, 900 K | Temperatura Joule `T = T0 + RTH·G·V²` y umbral de ruptura. |
| `THERMAL_DLAM_MAX` | 0.05 | Máximo cambio de `lam` por paso. |

Si en algún paso una juntura recorta su crecimiento a `THERMAL_DLAM_MAX`,
el modelo emite `StepSaturationWarning` una vez por corrida, con la
cantidad de junturas afectadas. Si son muchas, conviene reducir
`TIME_STEP_DT`. Con los parámetros por defecto y `V_INPUT = 3.6` V,
unas pocas junturas con caídas de tensión grandes saturan incluso con
`dt = 1e-5` s, porque el crecimiento es exponencial en el campo. En esas
junturas el recorte actúa como una tasa máxima efectiva.

> Los valores por defecto de estos tres modelos son órdenes de magnitud
> ilustrativos, no ajustados a mediciones.

## Ver también

- [`arquitectura.md`](arquitectura.md) — flujo de datos entre módulos.
- Código fuente: `src/neuromorphic/fisica/evolvers/`. El registro está en
  `base.py` y cada modelo en su propio módulo (`stochastic1.py`,
  `stochastic2.py`, `ladder.py`, `ladder_numba.py`, `thermal.py`).