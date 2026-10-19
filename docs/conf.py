"""Minimal Sphinx documentation configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
project = "FunctionalMarket-Geometry"
author = "FunctionalMarket-Geometry contributors"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
master_doc = "index"
exclude_patterns = ["_build"]
html_theme = "alabaster"
