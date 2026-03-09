# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from sphinx_gallery.sorting import FileNameSortKey

import pyclf

sys.path.insert(0, os.path.abspath("../.."))

project = 'pyclf'
copyright = '2026, Simon Kojima'
author = 'Simon Kojima'
release = pyclf.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_multiversion",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {"image_light": "logo-light.png", "image_dark": "logo-dark.png"},
    "navbar_end": ["theme-switcher", "version-switcher"],
    "switcher": {
        "json_url": "https://simonkojima.github.io/pyclf-docs/versions.json",
        "version_match": release,
    },
}

html_static_path = ['_static']

html_context = {
    "version_match": release,
}
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"example_",
    "within_subsection_order": FileNameSortKey,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}