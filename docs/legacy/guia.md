# neuromorphicNWLamas — Guía histórica

> ⚠️ **Deprecado.** Este documento describe la API previa al empaquetado
> (uso de `config.ini`, imports planos, recarga de módulos con
> `importlib.reload`, funciones `build_graph2`/`find_electrode_nodes2`).
> La versión vigente está en [`arquitectura.md`](arquitectura.md). Para
> el sistema de modelos de memristor, ver [`evolvers.md`](evolvers.md).

---

# Simulador de Redes de Nanohilos Neuromórficas (Nanowire Networks - NWN)

Este repositorio contiene una librería modular en Python para la generación estocástica, modelado topológico y simulación eléctrica/dinámica de redes de nanohilos de plata (Ag). El sistema simula el comportamiento de "memristores" estocásticos autoensamblados en los puntos de intersección de los filamentos, emulando la plasticidad sináptica y los procesos de memoria del cerebro humano.

---

## 🗺️ Arquitectura de la Librería y Flujo de Datos

El código se encuentra completamente desacoplado bajo una arquitectura centralizada basada en un archivo de configuración. El flujo de datos sigue de manera estricta el siguiente camino:

[ config.ini ]  <-- (Archivo de texto con los parámetros físicos y numéricos)
│
▼
[ load_config.py ]  <-- (Lee el .ini, calcula variables físicas como R con Pi)
│
├───────────────────────┬───────────────────────┬───────────────────────┐
▼                       ▼                       ▼                       ▼
[ geometria.py ]        [ grafo.py ]            [ fisica.py ]          [ visualizacion.py ]
(Disposición espacial   (Topología de red,      (Matriz dispersa Y,    (Gráficas y análisis
de los nanohilos)       nodos y electrodos)     bucle estocástico)     de resultados)
│                       │                       │                       │
└───────────────────────┼───────────────────────┘                       │
▼                                               │
[ simulador.py ]  <-- (Orquesta el bucle temporal)      │
│                                               │
└───────────────────────────────────────────────┤
▼
[ Interfaz Colab / Main ]


### Los Módulos del Sistema:
1. **`config.ini`**: El archivo maestro de texto plano. Es el único lugar donde modificas las constantes del sistema (densidad de hilos, voltajes, dimensiones, paso temporal $dt$, etc.).
2. **`load_config.py`**: El traductor físico. Lee el archivo `.ini`, calcula dinámicamente constantes complejas utilizando fórmulas geométricas (como la resistencia de los cables en base a su sección cilíndrica y la constante $\pi$) y exporta un diccionario global `p`.
3. **`geometria.py`**: Controla el modelado físico y espacial de la deposición de nanohilos en el plano 2D, calculando analíticamente los puntos de cruce.
4. **`grafo.py`**: Transforma la red geométrica en una abstracción matemática basada en grafos (`networkx`) mediante un algoritmo exacto de duplicación de nodos.
5. **`fisica.py`**: Construye las ecuaciones circuitales matriciales (Kirchhoff) y actualiza estocásticamente las conductancias locales.
6. **`visualizacion.py`**: Concentra todas las herramientas gráficas y de ploteo científico para aislar la lógica numérica de la visual.
7. **`simulador.py`**: El motor temporal maestro que integra todos los módulos anteriores para correr experimentos físicos complejos.

---

## 🛠️ Documentación Detallada de Funciones

*Nota: Siguiendo los estándares de transparencia de datos (Enfoque A), las funciones extraen los parámetros estáticos directamente del diccionario centralizado `p`, manteniendo las firmas de las funciones limpias y limitadas únicamente a los objetos dinámicos de la simulación.*

### 📦 1. Módulo: `geometria.py`
Se encarga de la generación espacial estocástica de los nanocables sobre el sustrato y de la detección analítica de colisiones geométricas.

#### `generate_and_find_junctions()`
Genera una distribución uniforme de hilos con centros aleatorios y ángulos de orientación $\theta \in [0, \pi)$ utilizando las constantes `NUM_WIRES`, `LENGTH` y `AREA` del archivo de configuración. Luego, aplica un algoritmo analítico para encontrar los cruces de segmentos.
* **Entradas:** Ninguna (Lee internamente del diccionario de configuración).
* **Salidas:**
    * `wires` *(list)*: Diccionarios con los puntos extremos `p1` y `p2` de cada hilo.
    * `junctions` *(list)*: Diccionarios con el `id` de la juntura, posición exacta `pos` $[x, y]$ e índices de los dos hilos involucrados.
    * `wire_to_junctions` *(dict)*: Mapa de adyacencia que asocia cada hilo con todas las junturas que posee.

---

### 📦 2. Módulo: `grafo.py`
Mapea la estructura geométrica a un objeto de la librería `networkx` implementando una **topología por duplicación de nodos**.

#### `build_graph2(wires, junctions, wire_to_junctions)`
Construye el grafo de admitancia. Por cada juntura física, genera dos nodos en el grafo (uno por cada cara del hilo que cruza). La arista que une estos nodos duplicados representa la juntura memristiva y se inicializa en conductancia `G_OFF`. Los segmentos internos del nanohilo que conectan una juntura con otra se representan como aristas resistivas lineales ordinarias.
* **Entradas:** `wires` *(list)*, `junctions` *(list)*, `wire_to_junctions` *(dict)*.
* **Salidas:** `G` *(nx.Graph)*: Grafo topológico estructural.

#### `find_electrode_nodes2(G)`
Identifica qué nodos del grafo están lo suficientemente cerca de las fronteras izquierda ($x=0$) y derecha ($x=\text{AREA}$) basándose en el parámetro de borde `PROXIMITY_THRESHOLD` para actuar como terminales eléctricos (Ánodo/Cátodo).
* **Entradas:** `G` *(nx.Graph)*.
* **Salidas:** `input_nodes` *(list)*, `output_nodes` *(list)*.

#### `check_percolation(G, input_nodes, output_nodes)`
Verifica si existe conectividad topológica estructural entre los electrodos de entrada y salida utilizando algoritmos de búsqueda en grafos.

#### `get_shortest_path_length(G, input_nodes, output_nodes)`
Calcula la longitud topológica (número de saltos/nodos) del camino más corto que conecta los extremos a través de la red.

---

### 📦 3. Módulo: `fisica.py`
Resuelve la física de transporte eléctrico aplicando conservación de la carga en los nodos (Leyes de Kirchhoff) y leyes de conmutación probabilísticas.

#### `build_admittance_matrix2(G, input_nodes, output_nodes)`
Ensambla la matriz de admitancia del circuito disperso ($Y$) y el vector de corrientes externas ($I$) para plantear el sistema de ecuaciones lineales $Y \cdot V = I$.
* Utiliza el valor exacto de `R_WIRE_PER_LENGTH` calculado dinámicamente en el lector de configuración.
* Aplica condiciones de contorno de Dirichlet fijando los voltajes límite (`V_INPUT` y `V_GROUND`) en los electrodos e inyecta una conductancia de fuga mínima (`G_LEAK = 1e-12`) para evitar singularidades matemáticas.
* **Salidas:** Matriz CSR dispersa `Y`, vector `I_vec` y el diccionario de indexación de nodos `node_to_index`.

#### `update_stochastic_conductance2(G, V_solved, node_to_index)`
Aplica la dinámica de conmutación probabilística evaluando el módulo del voltaje local $|V_{\text{mem}}|$ en cada juntura memristiva:
1.  **Proceso SET (OFF $\rightarrow$ ON):** Si $|V_{\text{mem}}| > V_{\text{threshold}}$, la probabilidad de formación de filamentos aumenta exponencialmente guiada por los coeficientes `P0_SET` y `ALPHA_SET`.
2.  **Proceso RESET (ON $\rightarrow$ OFF):** Si el voltaje decae por debajo del umbral, el filamento colapsa de forma estocástica (relajación volátil) guiado por un factor de decaimiento térmico (`P_DECAY`) matizado por el estado de estabilidad del campo eléctrico local.
* **Entradas:** `G` *(nx.Graph)*, `V_solved` *(array)*, `node_to_index` *(dict)*.
* **Salidas:** Objeto `G` con los estados de conductancia actualizados.

#### `calculate_input_current(G, V_solved, node_to_index, input_nodes)`
Calcula la corriente neta total que sale del electrodo de entrada sumando las contribuciones individuales de corriente de todas las ramas frontera conectadas.

---

### 📦 4. Módulo: `visualizacion.py`
Módulo dedicado exclusivamente a la generación de gráficas de nivel científico, aislando el código visual de los núcleos numéricos.

#### `plot_simulation_results(history_time, history_G_total)`
Genera la gráfica de la evolución temporal de la conductancia de la red, reproduciendo el comportamiento dinámico de facilitación y relajación volátil (inspirado en la Fig. 2b del paper de referencia).
* Divide visualmente el gráfico en dos regiones mediante sombreados: Fase de Pulso (Potenciación) y Fase de Relajación (Decaimiento).
* **Entradas:** `history_time` *(list/array)*, `history_G_total` *(list/array)*.
* **Salidas:** `None` (Renderiza el gráfico directamente en pantalla).

---

### 📦 5. Módulo: `simulador.py`
Funciona como el orquestador del tiempo discreto ejecutando los bucles de simulación física.

#### `run_simulation_dynamic_pulse()`
Simula la respuesta temporal de la red ante un experimento de pulso y relajación (Fig. 2b). Somete al sistema a un voltaje alto (`V_INPUT`) durante un tiempo de estimulación y luego a un voltaje bajo (`V_READ`) para observar el decaimiento de memoria volátil. Modifica dinámicamente los estados de voltaje en el diccionario de configuración para que el motor numérico reaccione al vuelo.
* **Salidas:** Historiales de tiempo, conductancia total del sistema (en mS) y cantidad de junturas memristivas activas en estado `ON`.

---

## 🚀 Uso en Google Colab con Panel de Control

Para integrar y ejecutar este repositorio en Google Colab junto con una interfaz gráfica interactiva que permite modificar los parámetros sobre la marcha sin alterar el código base, sigue esta estructura de celdas:

### Celda 1: Clonar y Acceder al Repositorio
```python
import os
import sys

REPO_URL = "[https://github.com/tu_usuario/neuromorphicNWLamas.git](https://github.com/tu_usuario/neuromorphicNWLamas.git)"
REPO_NAME = REPO_URL.split("/")[-1].replace(".git", "")

if not os.path.exists(REPO_NAME):
    !git clone {REPO_URL}

os.chdir(REPO_NAME)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

print(f"✅ Entorno listo en: {os.getcwd()}")
Celda 2: Panel de Control Interactivo (UI)
Ejecuta este bloque para generar barras deslizantes que sobrescriben de forma automática tu archivo config.ini, recargan los módulos en caliente en la memoria de Python y grafican los resultados al presionar el botón:

Python
import configparser
import sys
import importlib
import ipywidgets as widgets
from IPython.display import display, clear_output
import visualizacion as vis

def guardar_y_recargar(b):
    with output_panel:
        clear_output(wait=True)
        print("💾 Guardando parámetros en config.ini...")
        
        # 1. Sobreescribir el config.ini local
        config = configparser.ConfigParser()
        config.read("config.ini")
        config['Network']['num_wires'] = str(w_num_wires.value)
        config['Network']['area'] = str(w_area.value)
        config['Electrical']['v_input'] = str(w_v_input.value)
        config['Electrical']['diametro_nm'] = str(w_diametro.value)
        
        with open("config.ini", "w") as configfile:
            config.write(configfile)
            
        # 2. Forzar recarga de módulos (Enfoque A)
        for modulo in ['load_config', 'geometria', 'grafo', 'fisica', 'visualizacion', 'simulador']:
            if modulo in sys.modules:
                importlib.reload(sys.modules[modulo])
                
        print("🔄 Parámetros cargados. Ejecutando el motor temporal...")
        
        # 3. Importar y Ejecutar
        from simulador import run_simulation_dynamic_pulse
        t_hist, g_hist, active_hist = run_simulation_dynamic_pulse()
        
        # 4. Graficar usando el módulo de visualización
        importlib.reload(vis)
        vis.plot_simulation_results(t_hist, g_hist)

# --- Renderizado de la Interfaz ---
cfg = configparser.ConfigParser()
cfg.read("config.ini")

w_num_wires = widgets.IntSlider(value=cfg.getint('Network', 'num_wires'), min=500, max=5000, step=100, description='Nº Hilos:')
w_area = widgets.FloatSlider(value=cfg.getfloat('Network', 'area'), min=500.0, max=2000.0, step=100.0, description='Área (µm):')
w_v_input = widgets.FloatSlider(value=cfg.getfloat('Electrical', 'v_input'), min=1.0, max=10.0, step=0.1, description='V Pulso (V):')
w_diametro = widgets.FloatSlider(value=cfg.getfloat('Electrical', 'diametro_nm'), min=50.0, max=200.0, step=5.0, description='Ø NW (nm):')

btn_correr = widgets.Button(description="🚀 Correr Simulación", button_style='success', layout=widgets.Layout(width='40%'))
btn_correr.on_click(guardar_y_recargar)
output_panel = widgets.Output()

display(widgets.HBox([widgets.VBox([w_num_wires, w_area]), widgets.VBox([w_v_input, w_diametro])]))
display(btn_correr, output_panel)
