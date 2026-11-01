import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "flownp-repro"
author = "Grupo del proyecto de IA Generativa (MUIA, UAM)"
language = "es"
extensions = ["sphinx.ext.mathjax", "sphinx_gallery.gen_gallery"]
html_theme = "pydata_sphinx_theme"
html_title = "Procesos neuronales con emparejamiento de flujo"
exclude_patterns = ["_build"]

sphinx_gallery_conf = {
    "examples_dirs": "../tutorials",
    "gallery_dirs": "tutoriales",
    "filename_pattern": r"plot_",
    "download_all_examples": False,
    "remove_config_comments": True,
}
