# Arquitectura

## Flujo de datos

~~~text
                 defaults.ini  (parámetros empaquetados)
                       │
                       ▼
                 load_config.py  ──►  simulation["parameters"]
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   geometria.py    grafo.py       fisica/      visualizacion.py
   (hilos y        (topología,    ├── admitancia.py   (gráficas)
    junturas)       electrodos)   ├── corrientes.py
                                   └── evolvers/
                                       └── EVOLVER_SPECS
                                           ◄── register_memristor_evol_model
        │              │              │              │
        └──────────────┴──────────────┘              │
                       ▼                             │
                 dinamica.py  ◄──────────────────────┘
                 (motor temporal)
                       │
                       ▼
            setup_simulation() / run_*
            (interfaz pública del paquete)
~~~

## Enfoque de parámetros

Todos los parámetros se cargan **una sola vez** en un diccionario
`simulation` con la estructura:

~~~python
simulation = {
    "parameters": {...},   # crudos + derivados
    "junctions": {...},    # geometría de la red
    "graph": <networkx.Graph>,
    "terminals": {...},    # electrodos de entrada/salida
    "circuit": {...},      # matriz Y, vector I, mapeo de nodos,
                           # arrays y máscaras por memristor
    "evolver_state": {...} # estado propio del modelo de evolución
}
~~~

Cada función recibe `simulation` y lee lo que necesita de ahí. No hay
parámetros globales ni leídos implícitamente del directorio de trabajo.
La única excepción de estado global es el generador de números
aleatorios: `setup_simulation` llama a `np.random.seed(RNG_SEED)` y
tanto la geometría como los evolvers consumen `numpy.random`. Los overrides se hacen vía `parms=` en `setup_simulation` o
`load_parameters`.

## Uso en Google Colab

~~~bash
pip install git+https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas.git
~~~

import configparser, importlib, sys
import ipywidgets as widgets
from IPython.display import display, clear_output

import neuromorphic
from neuromorphic import setup_simulation, run_simulation_dynamic_pulse
from neuromorphic.visualizacion import plot_simulation_results

# Panel interactivo: los sliders sobrescriben los parámetros en cada corrida.
w_num_wires = widgets.IntSlider(value=1300, min=800, max=3000, step=100, description='Nº hilos:')
w_area      = widgets.FloatSlider(value=1000.0, min=500.0, max=2000.0, step=100.0, description='Área (µm):')
w_v_input   = widgets.FloatSlider(value=3.6, min=1.0, max=10.0, step=0.1, description='V pulso (V):')
w_diametro  = widgets.FloatSlider(value=115.0, min=50.0, max=200.0, step=5.0, description='Ø NW (nm):')

btn = widgets.Button(description="Correr simulación", button_style='success')
out = widgets.Output()

def on_run(_):
    with out:
        clear_output(wait=True)
        sim = setup_simulation(parms={
            "NUM_WIRES": w_num_wires.value,
            "AREA": w_area.value,
            "V_INPUT": w_v_input.value,
            "DIAMETRO_NM": w_diametro.value,
        })
        t, g, _ = run_simulation_dynamic_pulse(sim)
        plot_simulation_results(sim, t, g)

btn.on_click(on_run)
display(widgets.HBox([widgets.VBox([w_num_wires, w_area]),
                      widgets.VBox([w_v_input, w_diametro])]))
display(btn, out)
~~~

Con el paquete instalable, **no hace falta clonar ni manipular
`sys.path`**: `pip install` trae el paquete y el `defaults.ini`. Los
tests, la documentación y los benchmarks no se instalan; para eso hay
que clonar el repositorio. Los overrides van todos por `parms=`.

## Registro de evolvers

`fisica/evolvers/base.py` mantiene un registro global
`EVOLVER_SPECS: dict[str, EvolverSpec]`, donde cada `EvolverSpec` agrupa
la función de update, una función de inicialización y los parámetros
propios del modelo con sus valores por defecto. Cada modelo se registra al
importar su módulo mediante el decorador
`register_memristor_evol_model("nombre", init=..., parameters=...)`.

Los modelos incluidos en el paquete se descubren solos: al importarse,
`fisica/evolvers/__init__.py` recorre con `pkgutil.iter_modules` los
módulos del directorio e importa con `importlib` todos los públicos
(salvo `base`), en orden alfabético. La lista queda en
`neuromorphic.fisica.evolvers.MODEL_MODULES`. Los módulos que empiezan
con `_` no se importan automáticamente; ahí van las utilidades
compartidas y el código con dependencias opcionales, como el kernel de
`ladder_numba`, que recién se carga al inicializar ese modelo.

El decorador está reexportado en `neuromorphic.fisica`, de modo que
`from neuromorphic.fisica import register_memristor_evol_model` es la
forma recomendada de importarlo. Registrar un nombre ya existente lo
sobrescribe sin aviso.

`initialize_evolver(simulation)` resuelve `p["EVOLVER"]` contra el
registro, completa los parámetros por defecto del modelo, recrea
`simulation["evolver_state"]` y llama a `init`. La invocan
`setup_simulation` y, al comienzo de cada corrida, el motor temporal. La clave se lee del `.ini` como `evolver_model` en la sección
`[Memristor]`, y puede sobrescribirse con `parms={"EVOLVER": "..."}`.

Ver [`evolvers.md`](evolvers.md) para el contrato completo, la lista de
modelos incluidos y ejemplos de implementaciones propias.