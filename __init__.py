"""
eazy_sed_fitting

Official eazy-py SED Fitting and Photometric Redshifts
---------------------------------------------------------

Wraps the official eazy-py pipeline (Brammer, van Dokkum, & Coppi 2008;
eazy-py 0.8.6) behind one config + one photometry CSV: the package
generates the catalog, translate file, FILTER.RES, and template list into
a self-contained run directory, executes the official fit, and writes
summary/array products (see ``results.py``).

Quick start (from the directory containing this package; templates
default to the packaged Brown et al. 2014 atlas):

    python -m eazy_sed_fitting fit --phot-csv sed_input.csv \\
        --z-min 0.05 --z-max 0.16 --z-step 0.001 --z-step-type linear \\
        --output-dir runs/target

    from eazy_sed_fitting import FitConfig, run_fit
    cfg = FitConfig(name="target")
    result = run_fit(cfg, "sed_input.csv", run_dir="runs/target")

Requirements:
  - numpy, scipy, astropy, matplotlib, eazy-py (see environment.yml)
"""

from .config import FitConfig, load_config
from .data import load_photometry, prepare_photometry
from .filters import available_filters, make_spherex_tophat
from .templates import prepare_templates_param
from .fitting import run_fit, build_run_dir, build_eazy_params
from .results import (FitResult, extract_sed, load_run, percentiles_from_lnp,
                      summarize)
from .plots import generate_plots, plot_sed, plot_zscan

__all__ = [
    "FitConfig",
    "load_config",
    "load_photometry",
    "prepare_photometry",
    "available_filters",
    "make_spherex_tophat",
    "prepare_templates_param",
    "run_fit",
    "build_run_dir",
    "build_eazy_params",
    "FitResult",
    "extract_sed",
    "load_run",
    "percentiles_from_lnp",
    "summarize",
    "generate_plots",
    "plot_sed",
    "plot_zscan",
]
