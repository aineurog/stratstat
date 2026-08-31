"""Sphinx configuration for StratStat."""

from __future__ import annotations

import sys
from pathlib import Path

# -- Path setup ---------------------------------------------------------------
# So that autodoc can import stratstat without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import stratstat  # noqa: E402

# -- Project information ------------------------------------------------------
project = "StratStat"
copyright = "2026, Syed Qaisar Jalil"
author = "Syed Qaisar Jalil"
version = stratstat.__version__
release = stratstat.__version__

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "myst_parser",
]

# Enable LaTeX math delimiters: \(...\) inline, \[...\] display
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
    "special-members": "__init__",
}
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = []
language = "en"

# -- Options for HTML output --------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_title = "StratStat"
html_short_title = "StratStat"
html_show_sourcelink = True

# -- Intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
