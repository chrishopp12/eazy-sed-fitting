"""
fitting.py

eazy-py Run Construction and Execution
---------------------------------------------------------

The one entry point is ``run_fit(config, phot, run_dir)``:

  1. builds a self-contained run directory (catalog, translate file,
     FILTER.RES, templates.param, template-error curve, config echo);
  2. constructs ``eazy.photoz.PhotoZ`` from the bundled parameter defaults
     plus this package's overrides (all paths absolute, so nothing depends
     on the working directory);
  3. runs the official fit: ``fit_catalog`` (all-template NNLS photo-z),
     optionally ``fit_single_templates`` (mode "single") and a fixed-z
     evaluation via ``fit_at_zbest(zbest=...)``;
  4. captures the PhotoZ state into a ``FitResult`` and writes the
     package's output products (``results.save_outputs``).

eazy-py's ``standard_output`` is never called (see ``results.py``), and
``FIX_ZSPEC`` stays off -- fixed-redshift fits always go through
``fit_at_zbest``, which requires the redshift to lie strictly inside the
grid (eazy silently skips edge values, leaving zeroed coefficients).

Data products (per run directory):
  catalog.csv         wide-format eazy catalog (id, f_<band>, e_<band>; uJy)
  zphot.translate     column -> filter-number mapping
  FILTER.RES(.info)   generated filter file (see filters.py)
  templates.param     generated template list (directory mode)
  template_error.dat  the TEF curve used
  config.json         FitConfig echo
  zphot.param.echo    resolved eazy parameters
  summary.csv / arrays.npz [/ singles.csv]   (see results.py)

Requirements:
  - numpy, astropy, eazy
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.table import Table

from .config import BASE_EAZY_PARAMS, DEFAULT_TEF_FILE, FitConfig, MISSING_FLUX
from .data import band_metadata, object_ids, prepare_photometry
from .filters import build_filter_res, write_translate
from .results import (FitResult, Z_PERCENTILES, extract_sed,
                      percentiles_from_lnp, save_outputs)
from .templates import prepare_templates_param

DEFAULT_OUTPUT_ROOT = Path("eazy_output")


@dataclass
class RunPaths:
    """Absolute paths of the generated inputs for one run."""
    run_dir: Path
    catalog: Path
    translate: Path
    filter_res: Path
    templates_param: Path
    tef_file: Path
    config_echo: Path
    param_echo: Path


# ------------------------------------
# Run-directory construction
# ------------------------------------

def build_run_dir(config: FitConfig, phot: Table, run_dir) -> RunPaths:
    """Write every input eazy needs into ``run_dir`` and return the paths.

    ``phot`` must already be policy-applied (``data.prepare_photometry``).
    """
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # A stale FILTER.RES.npy sidecar would silently override the text file.
    stale = run_dir / "FILTER.RES.npy"
    if stale.exists():
        stale.unlink()
        print(f"removed stale {stale.name}")

    band_numbers = build_filter_res(
        band_metadata(phot), config=config, res_path=run_dir / "FILTER.RES")
    write_translate(band_numbers, run_dir / "zphot.translate")
    _write_catalog(phot, band_numbers, run_dir / "catalog.csv")

    tef_src = Path(config.tef_file).expanduser() if config.tef_file else DEFAULT_TEF_FILE
    tef_dst = run_dir / "template_error.dat"
    shutil.copyfile(tef_src, tef_dst)

    templates_param = prepare_templates_param(config, run_dir)
    config.to_json(run_dir / "config.json")

    return RunPaths(
        run_dir=run_dir,
        catalog=run_dir / "catalog.csv",
        translate=run_dir / "zphot.translate",
        filter_res=run_dir / "FILTER.RES",
        templates_param=templates_param,
        tef_file=tef_dst,
        config_echo=run_dir / "config.json",
        param_echo=run_dir / "zphot.param.echo",
    )


def _write_catalog(phot: Table, band_numbers: dict[str, int], catalog_path) -> None:
    """Write the wide-format eazy catalog (one row per object, uJy)."""
    ids = object_ids(phot)
    bands = list(band_numbers)
    catalog = Table()
    catalog["id"] = ids
    flux = {band: np.full(len(ids), MISSING_FLUX) for band in bands}
    err = {band: np.full(len(ids), MISSING_FLUX) for band in bands}
    for i, oid in enumerate(ids):
        sub = phot[np.asarray(phot["id"]) == oid]
        for row in sub:
            flux[str(row["band"])][i] = float(row["flux_uJy"])
            err[str(row["band"])][i] = float(row["flux_err_uJy"])
    for band in bands:
        catalog[f"f_{band}"] = flux[band]
        catalog[f"e_{band}"] = err[band]
    catalog.write(catalog_path, format="ascii.csv", overwrite=True)


def build_eazy_params(config: FitConfig, paths: RunPaths) -> dict:
    """The eazy parameter overrides for this run (keys uppercase).

    Layered onto eazy's bundled ``zphot.param.default`` by PhotoZ; the
    resolved set is echoed to ``zphot.param.echo`` after the fit.
    """
    linear = config.z_step_type == "linear"
    params = dict(BASE_EAZY_PARAMS)
    params.update({
        "CATALOG_FILE": str(paths.catalog),
        "FILTERS_RES": str(paths.filter_res),
        "TEMPLATES_FILE": str(paths.templates_param),
        "TEMP_ERR_FILE": str(paths.tef_file),
        "TEMP_ERR_A2": config.tef_scale if config.tef else 0.0,
        "SYS_ERR": config.sys_err,
        "Z_MIN": config.z_min,
        # np.arange excludes the endpoint on linear grids; pad by half a step.
        "Z_MAX": config.z_max + (config.z_step / 2.0 if linear else 0.0),
        "Z_STEP": config.z_step,
        "Z_STEP_TYPE": 0 if linear else 1,
        "FITTER": config.fitter,
        "MAIN_OUTPUT_FILE": str(paths.run_dir / config.name),
    })
    if config.prior:
        params["APPLY_PRIOR"] = "y"
        params["PRIOR_FILE"] = str(Path(config.prior_file).expanduser().resolve())
        params["PRIOR_FILTER"] = config.prior_filter
    for key, value in config.extra_params.items():
        params[str(key).upper()] = value
    return params


# ------------------------------------
# Fit execution
# ------------------------------------

def run_fit(config: FitConfig, phot, run_dir=None, *,
            grid_from: FitResult | None = None) -> FitResult:
    """Run the official eazy-py fit for one photometry set.

    Parameters
    ----------
    config : FitConfig
        The fit scenario; ``config.validate()`` is called first.
    phot : str, Path, or Table
        SED-input CSV (or equivalent table); the data policy is applied.
    run_dir : str or Path or None
        Run directory; None uses ``eazy_output/<config.name>/`` under the
        current directory. [default: None]
    grid_from : FitResult or None
        A same-session result whose template grid (bandpass integrals over
        z x template x filter) is reused, skipping the expensive build.
        Valid only when the band sequence, redshift grid, and template set
        are identical -- e.g. band subsets of one catalog expressed as
        missing values. Enforced, not assumed. [default: None]

    Returns
    -------
    result : FitResult
        Captured fit products; also written to the run directory.
    """
    config.validate()
    phot = prepare_photometry(phot, config=config)
    run_dir = Path(run_dir) if run_dir else DEFAULT_OUTPUT_ROOT / config.name
    paths = build_run_dir(config, phot, run_dir)
    params = build_eazy_params(config, paths)
    tempfilt = None
    if grid_from is not None:
        tempfilt = _reusable_tempfilt(config, grid_from,
                                      [str(b) for b in band_metadata(phot)["band"]])

    # eazy imports matplotlib.pyplot inside its fit/plot paths; force a
    # non-interactive backend before the first eazy import.
    import matplotlib
    matplotlib.use("Agg")
    from eazy.photoz import PhotoZ

    photz = PhotoZ(param_file=None,
                   translate_file=str(paths.translate),
                   zeropoint_file=None,
                   params=params,
                   load_prior=config.prior,
                   load_products=False,
                   n_proc=config.n_proc,
                   compute_tef_lnp=config.tef_lnp,
                   tempfilt=tempfilt)

    ids = object_ids(phot)
    if photz.NOBJ != len(ids) or photz.NFILT != len(set(phot["band"])):
        raise RuntimeError(
            f"catalog mismatch: eazy sees NOBJ={photz.NOBJ}, NFILT={photz.NFILT}; "
            f"expected {len(ids)} objects, {len(set(phot['band']))} filters")

    photz.fit_catalog(n_proc=config.n_proc, prior=config.prior, beta_prior=False)

    # Capture the photo-z state before any fixed-z refit overwrites it.
    zgrid = np.asarray(photz.zgrid, float)
    z_ml = np.asarray(photz.zml, float).copy()
    chi2_fit = np.asarray(photz.chi2_fit, float).copy()
    lnp = np.asarray(photz.lnp, float).copy() if hasattr(photz, "lnp") else None
    chi2_best = np.asarray(photz.chi2_best, float).copy()
    coeffs_best = np.asarray(photz.coeffs_best, float).copy()
    fmodel = np.asarray(photz.fmodel, float).copy()
    z_chi2 = zgrid[np.argmin(chi2_fit, axis=1)]
    _warn_grid_edges(ids, zgrid, z_ml, z_chi2)

    # Percentiles come from our trapezoidal CDF of eazy's own ln P(z);
    # PhotoZ.pz_percentiles is boundary-fragile (see percentiles_from_lnp).
    if lnp is not None:
        z_percentiles = percentiles_from_lnp(zgrid, lnp)
    else:
        print("WARNING: no lnp available; percentiles set to NaN")
        z_percentiles = np.full((photz.NOBJ, len(Z_PERCENTILES)), np.nan)

    seds = [extract_sed(photz, i) if z_ml[i] > 0 else None for i in range(photz.NOBJ)]

    result = FitResult(
        config=config,
        run_dir=paths.run_dir,
        ids=ids,
        bands=[str(b) for b in band_metadata(phot)["band"]],
        template_names=[t.name for t in photz.templates],
        pivot=np.asarray(photz.pivot, float),
        zgrid=zgrid,
        fnu=np.asarray(photz.fnu, float).copy(),
        efnu=np.asarray(photz.efnu, float).copy(),
        ok_data=np.asarray(photz.ok_data, bool).copy(),
        nusefilt=np.asarray(photz.nusefilt, int).copy(),
        chi2_fit=chi2_fit,
        lnp=lnp,
        z_ml=z_ml,
        z_chi2=z_chi2,
        z_percentiles=z_percentiles,
        chi2_best=chi2_best,
        coeffs_best=coeffs_best,
        fmodel=fmodel,
        seds=seds,
        photz=photz,
    )

    if config.mode == "single":
        # No arguments: reuses the PhotoZ template grid, so the bandpass
        # integrals and IGM treatment match the combo fit exactly.
        _, ampl, singles_chi2, _ = photz.fit_single_templates(verbose=False)
        result.singles_chi2 = np.asarray(singles_chi2, float)
        result.singles_ampl = np.asarray(ampl, float)

    if config.save_zcoeffs and hasattr(photz, "fit_coeffs"):
        result.fit_coeffs = np.asarray(photz.fit_coeffs, float).copy()

    if config.z_fixed is not None:
        if not (zgrid[0] < config.z_fixed < zgrid[-1]):
            raise ValueError(
                f"z_fixed={config.z_fixed} is not strictly inside the realized "
                f"grid ({zgrid[0]:.4f}, {zgrid[-1]:.4f}); eazy would silently skip it")
        photz.fit_at_zbest(zbest=np.full(photz.NOBJ, config.z_fixed))
        result.z_fixed = float(config.z_fixed)
        result.chi2_fixed = np.asarray(photz.chi2_best, float).copy()
        result.coeffs_fixed = np.asarray(photz.coeffs_best, float).copy()
        result.fmodel_fixed = np.asarray(photz.fmodel, float).copy()
        result.seds_fixed = [extract_sed(photz, i, z=config.z_fixed)
                             for i in range(photz.NOBJ)]

    try:
        photz.param.write(str(paths.param_echo))
    except Exception as err:
        print(f"WARNING: could not write parameter echo: {err}")

    save_outputs(result)
    return result


def _reusable_tempfilt(config: FitConfig, grid_from: FitResult, bands: list[str]):
    """Validate and return a previous run's template grid for reuse.

    The grid holds the bandpass integral of every template at every grid
    redshift through every filter, so it transfers only between runs whose
    band sequence, redshift grid, and template set all match (fits that
    differ purely in which bands are marked missing).
    """
    if grid_from.photz is None:
        raise ValueError("grid_from carries no live PhotoZ handle (rehydrated run?)")
    same_setup = all(
        getattr(config, key) == getattr(grid_from.config, key)
        for key in ("z_min", "z_max", "z_step", "z_step_type",
                    "templates", "template_pattern"))
    if not same_setup or list(grid_from.bands) != list(bands):
        raise ValueError(
            "grid_from is not reusable: band sequence, redshift grid, or "
            "template settings differ from the previous run")
    print(f"reusing template grid from run {grid_from.config.name!r}")
    return grid_from.photz.tempfilt


def _warn_grid_edges(ids, zgrid, z_ml, z_chi2) -> None:
    """Flag failed or edge-pinned solutions (eazy's -1 sentinel is silent)."""
    for i, oid in enumerate(ids):
        if z_ml[i] <= 0:
            print(f"WARNING: object {oid!r}: z_ml sentinel {z_ml[i]:.1f} -- "
                  "ln P(z) peaks at the first grid point; widen the grid")
        elif np.isclose(z_chi2[i], zgrid[0]) or np.isclose(z_chi2[i], zgrid[-1]):
            print(f"WARNING: object {oid!r}: chi2 minimum on the grid edge "
                  f"(z={z_chi2[i]:.4f}); widen the grid")
