# neuromorphicNWLamas

Simulador estocástico de redes de nanohilos neuromórficas (Nanowire
Networks, NWN) en Python. Genera redes autoensambladas de nanohilos de
plata, las modela como grafos con junturas memristivas y resuelve su
dinámica eléctrica mediante las leyes de Kirchhoff, incluyendo efectos
de facilitación y relajación volátil inspirados en plasticidad sináptica.

---

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas.git
cd neuromorphicNWLamas
pip install -e .
```

Para usar el modelo de evolución `ladder_numba`:

```bash
pip install -e ".[numba]"
```

Para desarrollo (tests, lint, type-check):

```bash
pip install -e ".[dev]"
```

---

## Uso rápido

```python
from neuromorphic import setup_simulation, run_simulation_dynamic_pulse
from neuromorphic.visualizacion import plot_simulation_results

# Construye la simulación con los parámetros por defecto del paquete.
sim = setup_simulation()

# Corre el experimento de pulso y relajación (Fig. 2b del paper).
t, G_total, active = run_simulation_dynamic_pulse(sim)

# Grafica los resultados.
plot_simulation_results(sim, t, G_total)
```

### Overrides sin tocar archivos

Todas las variantes se controlan con un diccionario `parms` que se
aplica sobre los valores por defecto:

```python
sim = setup_simulation(parms={
    "NUM_WIRES": 1300,
    "T_PULSE": 5.0,
    "T_RELAX": 20.0,
    "TIME_STEP_DT": 1e-3,
})
```

Todas las claves de `parms` se incorporan a `simulation["parameters"]`,
incluidos arrays (por ejemplo, un valor por juntura para introducir
desorden). Las que no son claves del núcleo ni parámetros declarados por
un modelo de evolución emiten `UnknownParameterWarning`, como aviso ante
posibles errores de tipeo. Los valores derivados (`PROXIMITY_THRESHOLD`,
`R_WIRE_PER_LENGTH`, `TOTAL_TIME`) se recalculan automáticamente después
del merge.

### Inspección paso a paso

`run_simulation_dynamic_pulse` acepta un `callback(t, simulation, step_data)`
que se llama en cada paso, después de actualizar los memristores.
`step_data` trae el circuito resuelto en ese paso (`Y`, `I_vec`, `V_vec`),
el voltaje aplicado `V_input` y la conductancia equivalente `G_total` en
mS. Por ejemplo, para registrar la fracción de junturas activas y cortar
la corrida si la red satura:

```python
frac = []

def monitor(t, simulation, step_data):
    activos = simulation["circuit"]["memristor_active"]
    frac.append(activos.mean())
    if activos.all():
        raise StopIteration(f"red saturada en t={t:.3f} s")

try:
    run_simulation_dynamic_pulse(sim, callback=monitor)
except StopIteration as err:
    print(err)
```

Una excepción lanzada desde el callback interrumpe la corrida y se
propaga; el historial que devuelve la función se pierde, así que lo que
haga falta conservar debe guardarlo el propio callback.

### Reproducibilidad

`setup_simulation` siembra el generador global de NumPy con `RNG_SEED`
(clave `seed` de la sección `[Probabilities]` del `.ini`, 42 por
defecto) antes de generar la geometría. Para obtener otra realización de
la red basta con cambiar la semilla:

```python
sim = setup_simulation(parms={"RNG_SEED": 7})
```

### Configuración con archivo propio

```python
sim = setup_simulation(filepath="mi_config.ini")
```

El archivo `defaults.ini` empaquetado con la librería sirve de plantilla.

---

## Modelo físico

Cada nanohilo se representa como un segmento de longitud `L` con
resistencia distribuida. En cada cruce entre dos hilos se coloca un
**memristor** cuya conductancia conmuta estocásticamente entre dos
estados:

| Estado | Conductancia | Conmutación     |
|--------|--------------|-----------------|
| OFF    | `G_OFF`      | SET por voltaje |
| ON     | `G_ON`       | RESET por corriente |

El evolver `stochastic2` implementa las probabilidades reportadas en
Lamas et al. (2026):

- **SET**: P_set = Δt · P₀ · max(0, 1 − exp[−α(V_mem − V_th)])
- **RESET**: P_reset = min(1, Δt · P_decay · I_mem²)

donde I_mem = G_ON · V_mem es la corriente a través de la juntura. El
evolver por defecto, `stochastic1`, usa una variante anterior sin
dependencia en Δt ni en la corriente (ver
[`docs/evolvers.md`](https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas/blob/main/docs/evolvers.md)).

En cada paso temporal se resuelve el sistema Y · V = I sobre la matriz
de admitancia dispersa del grafo, aplicando condiciones de contorno de
Dirichlet en los electrodos.

La red presenta un **umbral de percolación** en torno a N ≈ 1200 hilos
para L = 70 µm y A = 1 mm². Por debajo de ese valor no existe camino
topológico entre electrodos y la simulación se rechaza con
`RuntimeError`.

---

## Modelos de evolución

El parámetro `evolver_model` del `.ini` (o `EVOLVER` en `parms=`) elige
la función que actualiza el estado de los memristores en cada paso
temporal. Por defecto se usa `stochastic1`.

Es posible registrar implementaciones propias sin tocar el paquete:

```python
from neuromorphic.fisica import register_memristor_evol_model

@register_memristor_evol_model("mi_modelo")
def mi_modelo(simulation, V_solved):
    # mutar simulation["circuit"]["memristor_g"]
    # y simulation["graph"].edges[u, v]["conductance"]
    return simulation["graph"]
```

```python
sim = setup_simulation(parms={"EVOLVER": "mi_modelo"})
```

Detalles del contrato, manejo del RNG y modelos incluidos en
[`docs/evolvers.md`](docs/evolvers.md).

---

## Estructura del paquete

```text
src/neuromorphic/
├── defaults.ini         # Parámetros por defecto (empaquetados)
├── load_config.py       # Carga y validación de parámetros
├── geometria.py         # Generación espacial de hilos y junturas
├── grafo.py             # Topología (networkx), electrodos, percolación
├── fisica/
│   ├── admitancia.py    # Matriz de admitancia dispersa Y
│   ├── corrientes.py    # Corrientes en los electrodos
│   ├── pulsos.py        # Formas de onda de excitación
│   └── evolvers/        # Registro de modelos de evolución de memristores
├── dinamica.py          # Motor temporal (pulso, relajación)
├── visualizacion.py     # Gráficas científicas
└── simulation.py        # Entry point: setup_simulation()

docs/                    # Arquitectura, evolvers y guía histórica
scripts/benchmarks.py    # Benchmarks de setup y hot loop
tests/                   # Suite de pytest
```

La API pública se expone desde el paquete:

```python
from neuromorphic import (
    setup_simulation,
    run_simulation_dynamic_pulse,
    plot_simulation_results,
    EVOLVER_SPECS,
)
```

Las piezas de bajo nivel (matriz de admitancia, corrientes, registro de
evolvers) están en `neuromorphic.fisica`:

```python
from neuromorphic.fisica import (
    build_admittance_matrix,
    calculate_input_current,
    calculate_output_current,
    register_memristor_evol_model,
)
```

---

## Tests

```bash
pytest
```

La suite corre en pocos segundos e incluye tests unitarios
por módulo e integración end-to-end con redes por encima y por debajo
del umbral de percolación.

---

## Benchmarks

Para medir el rendimiento del setup y del hot loop:

```bash
python scripts/benchmarks.py
```

Los valores por defecto (N=1500, seed=42) tardan menos de dos segundos
en total. Ajustá con `--n-wires`, `--steps` y `--repeat` según lo que
necesites.

---

## Referencia

Este código implementa el modelo descrito en:

> Lamas et al. (2026). *Stochastic Modeling of Silver Nanowire Networks
> for Neuromorphic Computing*. 55º Jornadas Argentinas de
> Informática (JAIiO).
> https://55jaiio.sadio.org.ar/wp-content/uploads/2026/07/151.pdf

---

## Documentación

La documentación completa del proyecto y de la API está disponible en
[https://qilpcm-iflp-conicet.github.io/neuromorphicNWLamas/]

---

## Licencia

MIT. Ver [LICENSE](LICENSE).

[![Tests](https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas/actions/workflows/tests.yml/badge.svg)](...)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](...)
