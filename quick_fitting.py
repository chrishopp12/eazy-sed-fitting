"""
quick_fitting.py

Vectorized Quick Engine (no eazy-py at runtime)
---------------------------------------------------------

``run_quick_fit(config, phot, run_dir)`` is a fast alternative to
``fitting.run_fit`` that takes the same inputs and writes the same
outputs, but never imports eazy-py: it re-derives eazy's likelihood with
vectorized numpy/scipy.

Why it exists: eazy-py precomputes a template grid -- every template
integrated through every filter at every grid redshift -- with
per-combination Python machinery and a multiprocessing pool, which takes
tens of minutes for a many-channel catalog fit against a large template
atlas. When the filter set changes from run to run (the SPHEREx case:
channels are per-object tophats, so each object is its own run) that
cost recurs every time. This engine flattens all filter curves into one
wavelength array, interpolates each template onto it once per redshift,
and recovers the per-band photon-counting integrals with trapezoid
weights and ``np.add.reduceat`` -- the same grid in seconds.

Reproduced exactly (eazy-py 0.8.6 conventions, verified against its
source):
  - the redshift grid: linear ``arange`` and log ``utils.log_zgrid``;
  - the SYS_ERR floor: ``efnu^2 = efnu_orig^2 + (SYS_ERR*max(fnu,0))^2``;
  - the template error function: a CubicSpline of the curve evaluated at
    ``pivot/(1+z)``, clipped to its endpoint values outside the nonzero
    range, scaled by TEMP_ERR_A2 (``eazy.templates.TemplateError``);
    per-band variance ``var = efnu^2 + (TEF*max(fnu,0))^2``;
  - NNLS with eazy's internal per-template renormalization
    (RENORM_TEMPLATES=y), coefficients returned in raw template units
    (``photoz.template_lsq``);
  - ln P(z) = -chi2/2 + tef_lnp, with ``tef_lnp = -0.5*sum(log var)``
    over valid bands, the 1100 A reddest-filter clip, and trapezoidal
    normalization (``compute_tef_lnp`` / ``compute_lnp``);
  - z_ml: the analytic 3-point parabola refinement of ln P(z), with the
    same -1 sentinel when the peak sits on the first grid point
    (``get_maxlnp_redshift``);
  - single mode: per-template unconstrained analytic amplitudes with the
    TEF applied, matching ``fit_single_templates`` (including its use of
    the unclamped flux in the TEF variance);
  - fixed-z evaluation strictly inside the grid (``fit_at_zbest``).

Known deviations (negligible for low-redshift work; use the official
engine when they matter):
  - No IGM absorption. eazy attenuates the template grid with
    Inoue et al. (2014) blueward of the Lyman limit; a warning is
    printed when the configured grid samples rest wavelengths where
    that would act.
  - Bandpass integrals are evaluated on the filter-curve wavelength
    nodes; eazy integrates on each template's wavelength grid. Agreement
    is at the sub-grid-step level, not bit-identical.
  - Best-fit and fixed-z designs are direct projections at the exact
    redshift; eazy interpolates its precomputed grid with a spline.
  - Arrays are float64 throughout (the official path is float64 too via
    the package's ARRAY_NBITS pin -- see ``config.BASE_EAZY_PARAMS``).
  - Priors and fitters other than "nnls" are not supported (hard error);
    ``config.extra_params`` (raw eazy parameter overrides) do not reach
    this engine and are ignored with a warning.

Data products (per run directory): config.json, catalog.csv,
templates.param, template_error.dat, engine.info, plus summary.csv /
arrays.npz [/ singles.csv] via ``results.save_outputs`` -- the same
products as an official run (which additionally writes the eazy input
files), so ``results.load_run`` and ``plots.generate_plots`` work on
either.

Requirements:
  - numpy, scipy, astropy
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import nnls

from .config import DEFAULT_TEF_FILE, FitConfig
from .data import band_metadata, is_spherex, object_ids, prepare_photometry, valid_rows
from .filters import get_filter_curve, make_spherex_tophat
from .fitting import DEFAULT_OUTPUT_ROOT, _warn_grid_edges, _write_catalog
from .results import FitResult, Z_PERCENTILES, percentiles_from_lnp, save_outputs
from .templates import prepare_templates_param

# eazy-py constants reproduced here (photoz.py / utils.py, 0.8.6).
CLIGHT_AA = 2.99792458e18   # speed of light [Angstrom/s] (utils.CLIGHT * 1e10)
CLIP_WAVELENGTH = 1100.0    # compute_lnp: P(z)=0 once this passes the reddest band
LNP_FLOOR = -1e20           # compute_lnp sentinel for non-finite ln P(z)
IGM_REST_LIMIT = 1300.0     # rest wavelength below which eazy's IGM factor departs from 1

# Quadrature node spacing target, lambda/dlambda. eazy integrates each band
# on the template's wavelength grid (~2.7 A for the bundled atlas in the
# optical); sampling only a curve's native nodes (20 A for some vendored
# curves) would miss template structure inside the band, so curves are
# refined to at least this resolution before building weights.
QUAD_RESOLUTION = 2000.0


# ------------------------------------
# eazy conventions: grid, TEF, quadrature
# ------------------------------------

def zgrid_from_config(config: FitConfig) -> np.ndarray:
    """The redshift grid the official engine would realize (float64).

    Mirrors ``PhotoZ.set_zgrid`` with the parameters ``build_eazy_params``
    sends: linear grids are ``arange(Z_MIN, Z_MAX + Z_STEP/2, Z_STEP)``
    (the half-step pad compensates arange's exclusive endpoint), log
    grids are eazy's ``utils.log_zgrid``.
    """
    if config.z_step_type == "linear":
        return np.arange(config.z_min, config.z_max + config.z_step / 2.0,
                         config.z_step, dtype=float)
    return np.exp(np.arange(np.log(1.0 + config.z_min),
                            np.log(1.0 + config.z_max), config.z_step)) - 1.0


def _trapz_dx(x: np.ndarray) -> np.ndarray:
    """Composite-trapezoid weights: ``dx @ y == trapezoid(y, x)``."""
    dx = np.zeros_like(x)
    diff = np.diff(x) / 2.0
    dx[:-1] += diff
    dx[1:] += diff
    return dx


class QuickTEF:
    """eazy's ``TemplateError``: spline at rest wavelength, clipped.

    ``tef(z)`` returns the fractional template error per band, the curve
    spline evaluated at ``pivot/(1+z)`` and scaled; outside the nonzero
    range of the curve the first/last nonzero values are used instead of
    extrapolating the spline.
    """

    def __init__(self, curve_file, pivot: np.ndarray, scale: float):
        curve = np.loadtxt(curve_file)
        self.te_x, self.te_y = curve[:, 0].astype(float), curve[:, 1].astype(float)
        self.pivot = np.asarray(pivot, float)
        self.scale = float(scale)
        nonzero = self.te_y > 0
        self.min_wavelength = self.te_x[nonzero].min()
        self.max_wavelength = self.te_x[nonzero].max()
        self.clip_lo = self.te_y[nonzero][0]
        self.clip_hi = self.te_y[nonzero][-1]
        self._spline = CubicSpline(self.te_x, self.te_y)

    def __call__(self, z: float) -> np.ndarray:
        rest = self.pivot / (1.0 + z)
        tef = self._spline(rest)
        tef[rest < self.min_wavelength] = self.clip_lo
        tef[rest > self.max_wavelength] = self.clip_hi
        return tef * self.scale


# ------------------------------------
# Projection channels and templates
# ------------------------------------

@dataclass
class Channels:
    """Flattened photon-counting projection channels for all bands.

    ``wave`` concatenates every band's filter-curve nodes (observed
    Angstrom); ``weight`` holds the matching trapezoid quadrature weights
    ``T/lambda * dlambda``, so a segment's weighted sum over a template's
    f_nu is the photon-counting bandpass average times ``wsum``.
    """
    wave: np.ndarray      # (M,) all bands' nodes, concatenated
    weight: np.ndarray    # (M,) T/lambda * trapezoid dlambda
    offsets: np.ndarray   # (NFILT,) segment starts for np.add.reduceat
    wsum: np.ndarray      # (NFILT,) per-band weight totals
    pivot: np.ndarray     # (NFILT,) pivot wavelengths [Angstrom]
    blue_min: float       # bluest nonzero-throughput node [Angstrom]


def _refine_curve(wave, thru, resolution: float = QUAD_RESOLUTION):
    """Subdivide curve segments to at least ``lambda/resolution`` spacing.

    A transmission curve is piecewise linear, so added nodes leave the
    filter itself unchanged -- they only give the trapezoid quadrature
    sample points between the native nodes, where the template spectrum
    has structure the native sampling would skip.
    """
    pieces = [wave[:1]]
    for i in range(len(wave) - 1):
        n_sub = int(np.ceil((wave[i + 1] - wave[i]) / (wave[i] / resolution)))
        if n_sub > 1:
            pieces.append(np.linspace(wave[i], wave[i + 1], n_sub + 1)[1:])
        else:
            pieces.append(wave[i + 1:i + 2])
    refined = np.concatenate(pieces)
    return refined, np.interp(refined, wave, thru)


def build_channels(band_meta, config: FitConfig) -> Channels:
    """Resolve every band to filter-curve nodes and quadrature weights.

    Broadband curves come from the vendored registry (or
    ``config.extra_filters``); SPHEREx channels are the same padded
    tophats ``filters.build_filter_res`` writes into FILTER.RES. Curves
    are refined to the quadrature resolution (``_refine_curve``), and the
    pivot wavelength uses eazy's definition,
    ``sqrt(trapz(T*lam) / trapz(T/lam))``.
    """
    waves, weights, wsums, pivots = [], [], [], []
    blue_min = np.inf
    for row in band_meta:
        band = str(row["band"])
        if is_spherex(band, config.spherex_prefix):
            wave, thru = make_spherex_tophat(row["wave_um"], row["bandwidth_um"])
        else:
            wave, thru = get_filter_curve(band, extra_filters=config.extra_filters)
        wave, thru = _refine_curve(wave, thru)
        weight = thru / wave * _trapz_dx(wave)
        waves.append(wave)
        weights.append(weight)
        wsums.append(weight.sum())
        pivots.append(np.sqrt(np.trapezoid(thru * wave, wave)
                              / np.trapezoid(thru / wave, wave)))
        blue_min = min(blue_min, float(wave[thru > 0].min()))
    lengths = [len(w) for w in waves]
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    return Channels(wave=np.concatenate(waves), weight=np.concatenate(weights),
                    offsets=offsets, wsum=np.asarray(wsums),
                    pivot=np.asarray(pivots), blue_min=blue_min)


def load_quick_templates(config: FitConfig, run_dir) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load the template set as ``(name, wave_rest, fnu_rest)`` triples.

    Resolution goes through ``templates.prepare_templates_param`` (so the
    template set and the run-dir provenance are identical to an official
    run), then each listed spectrum is read directly: two-column ASCII
    wavelength [Angstrom] / f_lambda, ``#`` comments allowed. Wavelengths
    are scaled by the ``.param`` file's conversion column, sorted, and
    de-duplicated; fluxes convert to f_nu with eazy's constant
    (``flam * wave**2 / CLIGHT_AA``) so fitted amplitudes match the
    official engine's coefficient scale.
    """
    param_path = prepare_templates_param(config, run_dir)
    templates = []
    for line in Path(param_path).read_text().splitlines():
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if len(tokens) < 2:
            raise ValueError(f"{param_path}: unreadable templates row {line!r}")
        spec_path = Path(tokens[1])
        if not spec_path.is_absolute():
            spec_path = Path(param_path).parent / spec_path
        wave_scale = float(tokens[2]) if len(tokens) > 2 else 1.0
        try:
            data = np.loadtxt(spec_path, comments="#", usecols=(0, 1))
        except Exception as err:
            raise ValueError(
                f"quick engine reads two-column ASCII spectra only; "
                f"{spec_path} failed ({err}); use the official engine for "
                f"FITS/CSV template tables") from None
        wave = data[:, 0] * wave_scale
        flam = data[:, 1]
        order = np.argsort(wave)
        wave, flam = wave[order], flam[order]
        keep = np.concatenate(([True], np.diff(wave) > 0))
        wave, flam = wave[keep], flam[keep]
        # eazy names templates by file basename, extension included.
        templates.append((spec_path.name, wave, flam * wave**2 / CLIGHT_AA))
    return templates


def design_matrix(z: float, templates, ch: Channels) -> np.ndarray:
    """Synthetic photometry of every template at one redshift.

    Each template's rest-frame f_nu is interpolated onto the flattened
    channel nodes at ``wave/(1+z)`` (zero outside its coverage, matching
    a zero-overlap bandpass) and reduced to per-band photon-counting
    averages. Returns ``(NTEMP, NFILT)``; no (1+z) flux factor is
    applied -- the fit amplitude absorbs it, exactly as in eazy.
    """
    rest = ch.wave / (1.0 + z)
    rows = np.empty((len(templates), len(ch.wsum)))
    for it, (_, wave, fnu) in enumerate(templates):
        y = np.interp(rest, wave, fnu, left=0.0, right=0.0)
        rows[it] = np.add.reduceat(y * ch.weight, ch.offsets) / ch.wsum
    return rows


def design_cube(zgrid, templates, ch: Channels) -> np.ndarray:
    """The full template grid, ``(NZ, NTEMP, NFILT)`` -- eazy's tempfilt."""
    cube = np.empty((len(zgrid), len(templates), len(ch.wsum)))
    for iz, z in enumerate(zgrid):
        cube[iz] = design_matrix(z, templates, ch)
    return cube


# ------------------------------------
# Fitting cores (eazy photoz.py equivalents)
# ------------------------------------

def _nnls_fit(design, fnu, var, ok_band):
    """One NNLS solve, mirroring ``template_lsq`` (fitter="nnls").

    Applies eazy's internal per-template renormalization (the
    RENORM_TEMPLATES=y default) for conditioning and divides it back out,
    so the returned coefficients are raw template amplitudes. Returns
    ``(chi2, coeffs, fmodel)`` with chi2 summed over valid bands and
    ``fmodel`` evaluated for every band.
    """
    rms = np.sqrt(var)
    anorm = np.linalg.norm((design / rms)[:, ok_band], axis=1)
    ok_temp = anorm > 0
    anorm[~ok_temp] = 1.0
    normed = design / anorm[:, None]
    coeffs = np.zeros(design.shape[0])
    if ok_temp.any():
        weighted = (normed / rms).T[ok_band, :]
        solution, _ = nnls(weighted[:, ok_temp], (fnu / rms)[ok_band])
        coeffs[ok_temp] = solution
    fmodel = coeffs @ normed
    chi2 = float((((fnu - fmodel) ** 2 / var)[ok_band]).sum())
    return chi2, coeffs / anorm, fmodel


def _fit_object_scan(cube, tefgrid, fnu, efnu, ok_band, *, want_coeffs=False):
    """chi2(z), tef_lnp(z), and optionally coeffs(z) for one object."""
    nz = cube.shape[0]
    chi2 = np.empty(nz)
    tef_lnp = np.empty(nz)
    coeffs = np.empty((nz, cube.shape[1])) if want_coeffs else None
    fpos = np.maximum(fnu, 0.0)
    for iz in range(nz):
        var = efnu**2 + (tefgrid[iz] * fpos)**2
        chi2[iz], coeffs_iz, _ = _nnls_fit(cube[iz], fnu, var, ok_band)
        tef_lnp[iz] = -0.5 * float(np.log(var[ok_band]).sum())
        if want_coeffs:
            coeffs[iz] = coeffs_iz
    return chi2, tef_lnp, coeffs


def _fit_singles_scan(cube, tefgrid, fnu, efnu, ok_band):
    """Per-template analytic scan, mirroring ``fit_single_templates``.

    Amplitudes are unconstrained (can be negative; ``results`` applies
    the physical > 0 rule when reporting) and the TEF enters through
    eazy's exact expression ``efnu^2 + (fnu*TEF)^2`` -- the flux is NOT
    clamped to positive here, faithfully reproducing the official
    routine's (minor) inconsistency with the combo path. Returns
    ``(chi2, ampl)`` of shape ``(NTEMP, NZ)``.
    """
    var = efnu[None, :]**2 + (tefgrid * fnu[None, :])**2        # (NZ, NFILT)
    ivar = np.where(ok_band[None, :], 1.0 / var, 0.0)
    num = np.einsum("zf,ztf->zt", ivar * fnu[None, :], cube)
    den = np.einsum("zf,ztf->zt", ivar, cube**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ampl = np.where(den > 0, num / den, 0.0)
    base = (ivar * fnu[None, :]**2).sum(axis=1)                 # chi2 of a zero model
    chi2 = np.where(den > 0, base[:, None] - ampl * num, base[:, None])
    return chi2.T, ampl.T


def _lnp_from_scan(chi2_fit, tef_lnp, zgrid, lc_reddest, *, use_tef_lnp):
    """ln P(z) per object, mirroring ``compute_lnp`` (no priors)."""
    loglike = -chi2_fit / 2.0
    if use_tef_lnp:
        loglike = loglike + tef_lnp
    clip = (CLIP_WAVELENGTH * (1.0 + zgrid))[None, :] > lc_reddest[:, None]
    loglike[clip] = -np.inf
    loglike[~np.isfinite(loglike)] = LNP_FLOOR
    lnpmax = loglike.max(axis=1)
    pz = np.exp(loglike - lnpmax[:, None])
    log_norm = np.log(pz @ _trapz_dx(zgrid))
    lnp = loglike - lnpmax[:, None] - log_norm[:, None]
    lnp[~np.isfinite(lnp)] = LNP_FLOOR
    return lnp


def _zml_parabola(zgrid, lnp) -> np.ndarray:
    """Maximum-likelihood redshifts, mirroring ``get_maxlnp_redshift``.

    A parabola through the three grid points around the ln P(z) peak
    gives a continuous z_ml; a peak on the first grid point returns the
    -1 failure sentinel and a peak on the last returns the grid value,
    exactly as in eazy.
    """
    z_ml = np.empty(lnp.shape[0])
    nz = len(zgrid)
    for i, row in enumerate(lnp):
        iz = int(np.argmax(row))
        if iz == 0:
            z_ml[i] = -1.0
            continue
        if iz >= nz - 1:
            z_ml[i] = zgrid[iz]
            continue
        x, y = zgrid[iz - 1:iz + 2], row[iz - 1:iz + 2]
        dx, dx2, dy = np.diff(x), np.diff(x**2), np.diff(y)
        c2 = (dy[1] / dx[1] - dy[0] / dx[0]) / (dx2[1] / dx[1] - dx2[0] / dx[0])
        refined = np.nan
        if np.isfinite(c2) and c2 != 0:
            c1 = (dy[0] - c2 * dx2[0]) / dx[0]
            refined = -c1 / (2.0 * c2)
        # A degenerate (flat) parabola falls back to the grid peak.
        z_ml[i] = refined if np.isfinite(refined) else zgrid[iz]
    return z_ml


def _sed_payload(z, chi2, coeffs, fmodel, fnu, efnu, templates, tempflux, wave0):
    """The per-object SED block, matching ``results.extract_sed``'s keys.

    Like eazy's ``show_fit``, the blended spectrum is rendered on the
    first template's wavelength grid (every template resampled onto it
    once, in ``tempflux``).
    """
    return {
        "templz": wave0 * (1.0 + z),
        "templf": coeffs @ tempflux,
        "model": fmodel.copy(),
        "fobs": fnu.copy(),
        "efobs": efnu.copy(),
        "z": float(z),
        "chi2": float(chi2),
    }


# ------------------------------------
# Run construction and execution
# ------------------------------------

def _validate_quick(config: FitConfig) -> None:
    """Reject configurations the quick engine cannot honor."""
    if config.prior:
        raise ValueError("the quick engine does not support priors; "
                         "run the official engine (drop --quick)")
    if config.fitter != "nnls":
        raise ValueError(f"the quick engine implements fitter='nnls' only, "
                         f"got {config.fitter!r}; run the official engine")
    if config.extra_params:
        print("WARNING: config.extra_params are eazy-py parameter overrides; "
              f"the quick engine ignores them: {sorted(config.extra_params)}")


def _build_quick_run_dir(config: FitConfig, phot, run_dir: Path) -> Path:
    """Write the quick run's provenance files; returns the TEF file path.

    The catalog, config echo, templates.param, and TEF curve match an
    official run directory; the eazy-only inputs (FILTER.RES, translate,
    parameter echo) are not needed and not written. ``engine.info`` marks
    the directory as a quick-engine product.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    bands = [str(b) for b in band_metadata(phot)["band"]]
    _write_catalog(phot, {band: i + 1 for i, band in enumerate(bands)},
                   run_dir / "catalog.csv")
    tef_src = Path(config.tef_file).expanduser() if config.tef_file else DEFAULT_TEF_FILE
    tef_dst = run_dir / "template_error.dat"
    shutil.copyfile(tef_src, tef_dst)
    config.to_json(run_dir / "config.json")
    (run_dir / "engine.info").write_text(
        "engine: quick (vectorized eazy-faithful reimplementation; no eazy-py)\n"
        "see quick_fitting.py for the fidelity notes\n")
    return tef_dst


def run_quick_fit(config: FitConfig, phot, run_dir=None) -> FitResult:
    """Run the vectorized quick fit for one photometry set.

    Parameters
    ----------
    config : FitConfig
        The fit scenario; ``validate()`` is called first, plus the
        quick-engine restrictions (nnls only, no priors).
    phot : str, Path, or Table
        SED-input CSV (or equivalent table); the data policy is applied.
    run_dir : str or Path or None
        Run directory; None uses ``eazy_output/<config.name>/`` under the
        current directory. [default: None]

    Returns
    -------
    result : FitResult
        The same container ``fitting.run_fit`` returns (``photz`` is
        None); outputs are written to the run directory.
    """
    config.validate()
    _validate_quick(config)
    phot = prepare_photometry(phot, config=config)
    run_dir = Path(run_dir).resolve() if run_dir else (DEFAULT_OUTPUT_ROOT / config.name).resolve()
    tef_file = _build_quick_run_dir(config, phot, run_dir)

    bands = [str(b) for b in band_metadata(phot)["band"]]
    channels = build_channels(band_metadata(phot), config)
    templates = load_quick_templates(config, run_dir)
    template_names = [name for name, _, _ in templates]
    zgrid = zgrid_from_config(config)

    if config.z_fixed is not None and not (zgrid[0] < config.z_fixed < zgrid[-1]):
        raise ValueError(
            f"z_fixed={config.z_fixed} is not strictly inside the realized "
            f"grid ({zgrid[0]:.4f}, {zgrid[-1]:.4f})")
    if zgrid[-1] > channels.blue_min / IGM_REST_LIMIT - 1.0:
        print(f"WARNING: the grid reaches z={zgrid[-1]:.2f}, where the bluest "
              f"band samples rest wavelengths below {IGM_REST_LIMIT:.0f} A; "
              "the quick engine applies no IGM absorption -- use the "
              "official engine for high-redshift fits")

    tef = QuickTEF(tef_file, channels.pivot,
                   scale=(config.tef_scale if config.tef else 0.0))
    tefgrid = np.array([tef(z) for z in zgrid])                  # (NZ, NFILT)
    cube = design_cube(zgrid, templates, channels)               # (NZ, NTEMP, NFILT)

    # Per-object photometry in catalog band order, eazy conventions:
    # efnu carries the SYS_ERR floor on positive fluxes, missing stays -99.
    ids = object_ids(phot)
    nobj, nfilt, ntemp, nz = len(ids), len(bands), len(templates), len(zgrid)
    fnu = np.empty((nobj, nfilt))
    efnu = np.empty((nobj, nfilt))
    ok_data = np.zeros((nobj, nfilt), bool)
    for i, oid in enumerate(ids):
        sub = phot[np.asarray(phot["id"]) == oid]
        ok = valid_rows(sub)
        raw_f = np.asarray(sub["flux_uJy"], float)
        raw_e = np.asarray(sub["flux_err_uJy"], float)
        fnu[i] = raw_f
        efnu[i] = np.where(ok, np.sqrt(raw_e**2 + (config.sys_err
                                                   * np.maximum(raw_f, 0.0))**2), raw_e)
        ok_data[i] = ok
    nusefilt = ok_data.sum(axis=1)
    lc_reddest = (ok_data * channels.pivot[None, :]).max(axis=1)

    chi2_fit = np.empty((nobj, nz))
    tef_lnp = np.empty((nobj, nz))
    fit_coeffs = np.empty((nobj, nz, ntemp)) if config.save_zcoeffs else None
    for i in range(nobj):
        chi2_fit[i], tef_lnp[i], coeffs_z = _fit_object_scan(
            cube, tefgrid, fnu[i], efnu[i], ok_data[i],
            want_coeffs=config.save_zcoeffs)
        if config.save_zcoeffs:
            fit_coeffs[i] = coeffs_z

    lnp = _lnp_from_scan(chi2_fit, tef_lnp, zgrid, lc_reddest,
                         use_tef_lnp=config.tef_lnp)
    z_ml = _zml_parabola(zgrid, lnp)
    z_chi2 = zgrid[np.argmin(chi2_fit, axis=1)]
    _warn_grid_edges(ids, zgrid, z_ml, z_chi2)
    z_percentiles = percentiles_from_lnp(zgrid, lnp)

    # Every template resampled onto the first one's grid, for SED rendering
    # (eazy show_fit does the same; plain interp extends the end values).
    wave0 = templates[0][1]
    tempflux = np.array([np.interp(wave0, wave, flux)
                         for _, wave, flux in templates])

    def evaluate_at(z, i):
        """Best-fit products for object ``i`` at exact redshift ``z``."""
        var = efnu[i]**2 + (tef(z) * np.maximum(fnu[i], 0.0))**2
        chi2, coeffs, fmodel = _nnls_fit(design_matrix(z, templates, channels),
                                         fnu[i], var, ok_data[i])
        sed = _sed_payload(z, chi2, coeffs, fmodel, fnu[i], efnu[i],
                           templates, tempflux, wave0)
        return chi2, coeffs, fmodel, sed

    chi2_best = np.zeros(nobj)
    coeffs_best = np.zeros((nobj, ntemp))
    fmodel = np.zeros((nobj, nfilt))
    seds = []
    for i in range(nobj):
        # eazy only refits redshifts strictly inside the grid; edge and
        # failed solutions keep zeroed best-fit products (and no SED).
        if zgrid[0] < z_ml[i] < zgrid[-1]:
            chi2_best[i], coeffs_best[i], fmodel[i], sed = evaluate_at(z_ml[i], i)
            seds.append(sed)
        else:
            seds.append(None)

    result = FitResult(
        config=config,
        run_dir=run_dir,
        ids=ids,
        bands=bands,
        template_names=template_names,
        pivot=channels.pivot,
        zgrid=zgrid,
        fnu=fnu,
        efnu=efnu,
        ok_data=ok_data,
        nusefilt=nusefilt,
        chi2_fit=chi2_fit,
        lnp=lnp,
        z_ml=z_ml,
        z_chi2=z_chi2,
        z_percentiles=z_percentiles,
        chi2_best=chi2_best,
        coeffs_best=coeffs_best,
        fmodel=fmodel,
        seds=seds,
        photz=None,
    )

    if config.mode == "single":
        singles_chi2 = np.empty((ntemp, nobj, nz))
        singles_ampl = np.empty((ntemp, nobj, nz))
        for i in range(nobj):
            singles_chi2[:, i, :], singles_ampl[:, i, :] = _fit_singles_scan(
                cube, tefgrid, fnu[i], efnu[i], ok_data[i])
        result.singles_chi2 = singles_chi2
        result.singles_ampl = singles_ampl

    if config.save_zcoeffs:
        result.fit_coeffs = fit_coeffs

    if config.z_fixed is not None:
        result.z_fixed = float(config.z_fixed)
        result.chi2_fixed = np.zeros(nobj)
        result.coeffs_fixed = np.zeros((nobj, ntemp))
        result.fmodel_fixed = np.zeros((nobj, nfilt))
        result.seds_fixed = []
        for i in range(nobj):
            (result.chi2_fixed[i], result.coeffs_fixed[i],
             result.fmodel_fixed[i], sed) = evaluate_at(config.z_fixed, i)
            result.seds_fixed.append(sed)

    save_outputs(result)
    return result
