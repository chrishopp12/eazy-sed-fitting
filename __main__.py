#!/usr/bin/env python3
"""
__main__.py

Command-Line Interface
---------------------------------------------------------

Runs an official eazy-py fit from a photometry CSV, regenerates figures
from a finished run directory, or lists the vendored filter set. Flags
only override a loaded ``FitConfig``; the full scenario lives in the
config JSON (``FitConfig.to_json``), which each run echoes into its run
directory. ``--quick`` swaps in the vectorized quick engine
(``quick_fitting.py``): same inputs, same outputs, seconds instead of a
long template-grid build, no eazy-py import.

Requirements:
  - numpy, astropy, matplotlib, eazy-py (official fit only)

Usage:
  python -m eazy_sed_fitting fit --phot-csv PHOT.csv [--quick]
      [--config CFG.json] [--output-dir DIR] [--name TAG]
      [--mode combo|single] [--templates PATH]
      [--z-min F --z-max F --z-step F]
      [--z-step-type linear|log] [--z-fixed F] [--sys-err F] [--no-tef]
      [--min-bands N] [--min-snr-broadband F] [--n-proc N] [--plots]
      [--z-ref F]
  python -m eazy_sed_fitting plot RUN_DIR [--z-ref F]
  python -m eazy_sed_fitting filters

Examples:
  Photo-z for one target with the packaged Brown+2014 atlas on a linear grid:
    python -m eazy_sed_fitting fit --phot-csv sed_input.csv \\
        --z-min 0.05 --z-max 0.16 --z-step 0.001 --z-step-type linear \\
        --name target1 --output-dir runs/target1 --plots
  Re-draw the figures of a finished run with a reference redshift:
    python -m eazy_sed_fitting plot runs/target1 --z-ref 0.106
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import load_config
from .fitting import run_fit
from .filters import available_filters
from .plots import generate_plots
from .quick_fitting import run_quick_fit
from .results import load_run, summarize

# CLI destination -> FitConfig field for the plain value overrides.
_OVERRIDE_FIELDS = {
    "name": "name",
    "mode": "mode",
    "z_fixed": "z_fixed",
    "z_min": "z_min",
    "z_max": "z_max",
    "z_step": "z_step",
    "z_step_type": "z_step_type",
    "templates": "templates",
    "template_pattern": "template_pattern",
    "sys_err": "sys_err",
    "tef_scale": "tef_scale",
    "min_bands": "min_valid_bands",
    "min_snr_broadband": "min_snr_broadband",
    "fitter": "fitter",
    "n_proc": "n_proc",
}


def _apply_overrides(config, args):
    """Layer the provided CLI flags onto a loaded config."""
    updates = {}
    for dest, fieldname in _OVERRIDE_FIELDS.items():
        value = getattr(args, dest, None)
        if value is not None:
            updates[fieldname] = value
    if args.no_tef:
        updates["tef"] = False
    if args.save_zcoeffs:
        updates["save_zcoeffs"] = True
    return replace(config, **updates) if updates else config


def _add_fit_parser(subparsers) -> None:
    p = subparsers.add_parser("fit", help="run an official eazy-py fit")
    p.add_argument("--phot-csv", required=True,
                   help="SED-input CSV (band, flux_uJy, flux_err_uJy, wave_um, bandwidth_um)")
    p.add_argument("--quick", action="store_true",
                   help="use the vectorized quick engine: same inputs and outputs, "
                        "no eazy-py grid build (see quick_fitting.py for fidelity notes)")
    p.add_argument("--config", default=None, help="FitConfig JSON to load before overrides")
    p.add_argument("--output-dir", "-o", default=None,
                   help="run directory [default: eazy_output/<name>]")
    p.add_argument("--name", default=None, help="run label / fallback object id")
    p.add_argument("--mode", choices=("combo", "single"), default=None,
                   help="combo (all-template NNLS) or single (adds per-template fits)")
    p.add_argument("--z-fixed", type=float, default=None,
                   help="also evaluate the best-fit SED at this redshift")
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--z-step", type=float, default=None,
                   help="grid step; fractional when --z-step-type log")
    p.add_argument("--z-step-type", choices=("linear", "log"), default=None)
    p.add_argument("--templates", default=None,
                   help="eazy templates .param file or a directory of spectra "
                        "[default: packaged Brown+2014 atlas]")
    p.add_argument("--template-pattern", default=None,
                   help="glob for directory-mode spectra [default: *_spec.dat]")
    p.add_argument("--sys-err", type=float, default=None,
                   help="fractional error floor (eazy SYS_ERR) [default: 0.05]")
    p.add_argument("--no-tef", action="store_true",
                   help="disable the template error function")
    p.add_argument("--tef-scale", type=float, default=None,
                   help="TEMP_ERR_A2 multiplier on the TEF curve [default: 1.0]")
    p.add_argument("--min-bands", type=int, default=None,
                   help="minimum valid bands per object [default: 5]")
    p.add_argument("--min-snr-broadband", type=float, default=None,
                   help="mark non-SPHEREx bands below this S/N missing [default: off]")
    p.add_argument("--fitter", default=None, help="eazy template solver [default: nnls]")
    p.add_argument("--n-proc", type=int, default=None, help="worker processes [default: 4]")
    p.add_argument("--save-zcoeffs", action="store_true",
                   help="persist the full (NOBJ, NZ, NTEMP) coefficient cube")
    p.add_argument("--plots", action="store_true", help="write SED and z-scan figures")
    p.add_argument("--z-ref", type=float, default=None,
                   help="reference redshift drawn on the figures")


def cmd_fit(args) -> None:
    config = _apply_overrides(load_config(args.config), args)
    engine = run_quick_fit if args.quick else run_fit
    result = engine(config, args.phot_csv, run_dir=args.output_dir)
    summarize(result).pprint(max_width=200)
    if args.plots:
        for path in generate_plots(result, z_ref=args.z_ref):
            print(f"figure -> {path}")


def cmd_plot(args) -> None:
    result = load_run(args.run_dir)
    for path in generate_plots(result, z_ref=args.z_ref):
        print(f"figure -> {path}")


def cmd_filters(_args) -> None:
    for band in available_filters():
        print(band)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m eazy_sed_fitting",
        description="Official eazy-py SED fitting and photometric redshifts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_fit_parser(subparsers)

    p_plot = subparsers.add_parser("plot", help="regenerate figures from a run directory")
    p_plot.add_argument("run_dir", help="run directory written by a previous fit")
    p_plot.add_argument("--z-ref", type=float, default=None,
                        help="reference redshift drawn on the figures")

    subparsers.add_parser("filters", help="list the vendored filter bands")

    args = parser.parse_args()
    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "filters":
        cmd_filters(args)


if __name__ == "__main__":
    main()
