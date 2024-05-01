# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(
    0, os.path.abspath("../squishyplanet")
)  # If your modules are in a parent directory
sys.path.insert(
    0, os.path.abspath("../squishyplanet/engine")
)  # Add the path to the "engine" subdirectory


project = "squishyplanet"
copyright = "2024, Ben Cassese"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",  # allows Google style-guide docs to render more prettily
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_automodapi.automodapi",
    "myst_parser",
    "sphinxcontrib.video",
    # "sphinx.ext.pngmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "visualizations"]
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "squishyplanet"
html_static_path = ["_static"]

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/ben-cassese/squishyplanet",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": True,
    "show_prev_next": False,
}

html_context = {"default_mode": "light"}

html_sidebars = {
    "**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]
}
