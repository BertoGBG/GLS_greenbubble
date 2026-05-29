import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "GreenBubble"
copyright = "2024, GreenBubble Authors"
author = "Alberto Alamia"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # Google/NumPy docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",               # Markdown support
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = ["snakemake"]

bibtex_bibfiles = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pypsa":  ("https://pypsa.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "GreenBubble"

html_theme_options = {
    "repository_url": "https://github.com/BertoGBG/GLS_greenbubble",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": False,
    "show_navbar_depth": 2,
}

html_static_path = ["_static"]
