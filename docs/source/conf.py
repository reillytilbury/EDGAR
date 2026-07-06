import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# Automatically run sphinx-apidoc during sphinx-build (so it works seamlessly on Read the Docs!)
def run_apidoc(app):
    import sphinx.ext.apidoc

    cur_dir = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(cur_dir, "api")
    module_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "edgar"))
    template_dir = os.path.join(cur_dir, "_templates", "apidoc")

    # Arguments for sphinx-apidoc: output folder, package folder, overwrite force, separate page per module, and custom templates
    sphinx.ext.apidoc.main(
        [
            "-o",
            output_path,
            module_path,
            "-f",
            "-e",
            "--templatedir",
            template_dir,
        ]
    )


# Project information
project = "EDGAR"
copyright = "2026, EDGAR Contributors"
author = "EDGAR Contributors"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# HTML options
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]


def setup(app):
    app.connect("builder-inited", run_apidoc)
