# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "Sage"
copyright = "2026, Narenraju Nagarajan"
author = "Narenraju Nagarajan"
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "myst_parser",
    "autoapi.extension",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinxext.opengraph",
]

# -- Open Graph / social card settings --------------------------------------

ogp_site_url = "https://sage-gw.readthedocs.io/"
ogp_image = "https://sage-gw.readthedocs.io/en/latest/_static/rectangular_logo.png"
ogp_description_length = 200
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta name="twitter:card" content="summary_large_image">',
    '<meta name="google-site-verification" content="cz-DZtJgJScLi4A7MPQyCntXuotPFiQXwh3g5uv3j-w">',
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Suppress nitpicky warnings on missing cross-references (common with torch types)
nitpicky = False

# Show full qualified names in signatures
add_module_names = False

# Type-hints rendered as inline text in the description, not in the signature
autodoc_typehints = "description"

# Napoleon settings (Google-style + NumPy-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_rtype = True

# AutoAPI settings
autoapi_type = "python"
autoapi_dirs = ["../../sage"]
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"

# Exclude internal / legacy / experimental sub-trees from the API docs
autoapi_ignore = [
    "*/evomcts/*",
    "*/legacy/*",
    "*/factory/legacy*",
    "*/__pycache__/*",
]

# Intersphinx mapping for cross-referencing external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_logo = "_static/rectangular_logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "display_github": True,
    "github_user": "nnarenraju",
    "github_repo": "sage",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
