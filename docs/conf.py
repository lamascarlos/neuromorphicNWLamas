# docs/conf.py
"""Configuración de Sphinx para neuromorphicNWLamas."""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: permite que autodoc importe `neuromorphic` desde src/
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Metadatos del proyecto
# ---------------------------------------------------------------------------
project = "neuromorphicNWLamas"
author = "QILPCM-IFLP-CONICET"
copyright = "2026, QILPCM-IFLP-CONICET"

try:
    from neuromorphic import __version__ as _v

    version = _v
    release = _v
except Exception:
    version = "0.1.0"
    release = "0.1.0"

# ---------------------------------------------------------------------------
# Extensiones
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # extrae docstrings de la API
    "sphinx.ext.napoleon",  # parsea docstrings NumPy/Google
    "sphinx.ext.viewcode",  # enlaces al código fuente
    "sphinx.ext.intersphinx",  # enlaces cruzados a numpy, scipy, networkx
    "myst_parser",  # soporte Markdown
    "sphinx_copybutton",  # botón "copiar" en bloques de código
    "sphinx_design",  # componentes visuales (grids, cards, tabs)
]

# Archivos fuente: .rst y .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "legacy/**",  # documentación histórica, no entra al build
]

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_mock_imports = []  # todas las deps están en el env de docs

# ---------------------------------------------------------------------------
# napoleon (docstrings NumPy)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# ---------------------------------------------------------------------------
# MyST (Markdown)
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",  # ::: para admonitions
    "deflist",  # listas de definiciones
    "fieldlist",  # :field: value
    "dollarmath",  # $...$ y $$...$$
    "amsmath",  # entornos \begin{align}
    "tasklist",  # checkboxes
    "attrs_block",  # atributos en bloques
]
myst_heading_anchors = 3  # anchors automáticos hasta h3

# ---------------------------------------------------------------------------
# intersphinx: referencias cruzadas a otras librerías
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = f"neuromorphicNWLamas {version}"
html_static_path = ["_static"]
html_logo = None  # agregar si tienen logo
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas/",
    "source_branch": "main",
    "source_directory": "docs/",
}

suppress_warnings = ["myst.header"]
