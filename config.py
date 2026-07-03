"""
config.py

Fit Configuration for the eazy-py Wrapper
---------------------------------------------------------

A fit scenario is described by a ``FitConfig``: the template set, redshift
grid, error model (SYS_ERR floor + template error function), fit mode
(all-template combination, optionally augmented by per-template single
fits), an optional fixed redshift, and the data policy (minimum band
count, broadband S/N cut).  ``fitting.run_fit`` consumes a config plus a
photometry CSV, so one codebase fits any target by swapping the config.

Target-specific configs are not hardcoded here.  Build a ``FitConfig`` and
serialize it next to the target's data (``cfg.to_json(path)``); load it
with ``load_config(path)`` or ``--config path`` on the CLI.

``BASE_EAZY_PARAMS`` pins the eazy-py parameters this package relies on:
catalog fluxes in microJansky (PRIOR_ABZP = 23.9), no Milky Way extinction
correction, and FIX_ZSPEC off -- fixed-redshift fits go through
``fit_at_zbest(zbest=...)`` instead, so a z_spec column can never silently
hijack the fit.

Notes:
  - eazy-py 0.8.6 silently ignores N_MIN_COLORS (its hard floor is a
    2-band skip); ``min_valid_bands`` is enforced in ``data.py`` instead.
  - eazy-py discards the template-file age column, so SSP template sets
    fit as plain template lists with no age-vs-universe cut (see
    ``templates.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

# ------------------------------------
# Package data locations
# ------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_DATA_DIR = PACKAGE_DIR / "data"
FILTER_DATA_DIR = PACKAGE_DATA_DIR / "filters"
DEFAULT_TEF_FILE = PACKAGE_DATA_DIR / "TEMPLATE_ERROR.eazy_v1.0"

SPHEREX_PREFIX = "SPHEREx_"

# Catalog value marking a band as not observed (below eazy's
# NOT_OBS_THRESHOLD default of -90).
MISSING_FLUX = -99.0

# ------------------------------------
# Invariant eazy-py parameter overrides
# ------------------------------------
# Applied to every run before the per-config values; config.extra_params
# is merged last and can override even these.

BASE_EAZY_PARAMS = {
    "PRIOR_ABZP": 23.9,        # flux of 1.0 in the catalog == AB 23.9 -> microJansky
    "CAT_HAS_EXTCORR": "y",    # with MW_EBV = 0: no Galactic extinction correction
    "MW_EBV": 0.0,
    "FIX_ZSPEC": "n",          # fixed-z fits use fit_at_zbest, never FIX_ZSPEC
    "APPLY_PRIOR": "n",        # flipped to "y" when config.prior is True
    "VERBOSITY": 1,
}


# ------------------------------------
# Fit configuration
# ------------------------------------

@dataclass
class FitConfig:
    """Full specification of one eazy-py fit.

    Serialize with ``to_json`` / load with ``from_json`` (or ``load_config``).
    ``validate()`` is called by ``fitting.run_fit``.

    Parameters
    ----------
    name : str
        Label, used for the run directory, output filenames, and the object
        id when the photometry CSV has no ``id`` column.
    mode : str
        "combo" (default): the official eazy fit, a non-negative
        least-squares combination of all templates. "single" additionally
        runs ``PhotoZ.fit_single_templates`` (each template fit alone on the
        redshift grid) and reports the best single template; the combo
        products are still computed.
    z_fixed : float or None
        If set, also evaluate the best-fit SED and coefficients at this
        redshift via ``fit_at_zbest`` (must lie strictly inside the grid).
        The photometric-redshift scan still runs. [default: None]
    z_min, z_max : float
        Redshift grid bounds. [default: 0.01, 6.0]
    z_step : float
        Grid step; fractional (dz = z_step * (1 + z)) when ``z_step_type``
        is "log", constant when "linear". [default: 0.01]
    z_step_type : str
        "log" (eazy default) or "linear". [default: "log"]
    templates : str
        Required. Path to an eazy templates ``.param`` file (used as-is;
        its internal paths must be absolute or eazy-resolvable), or to a
        directory of spectra gathered with ``template_pattern``.
    template_pattern : str
        Glob for directory-mode template spectra. [default: "*_spec.dat"]
    sys_err : float
        Fractional systematic error floor, added in quadrature to every
        band by eazy at catalog read time (SYS_ERR). [default: 0.05]
    tef : bool
        Apply the template error function. [default: True]
    tef_file : str or None
        Two-column TEF curve (rest wavelength Angstrom, fractional error);
        None uses the packaged ``TEMPLATE_ERROR.eazy_v1.0``. [default: None]
    tef_scale : float
        TEMP_ERR_A2 multiplier on the TEF curve; forced to 0.0 when ``tef``
        is False. [default: 1.0]
    tef_lnp : bool
        Include eazy's TEF Gaussian-normalization term in ln P(z)
        (``compute_tef_lnp``); the official behavior. [default: True]
    prior : bool
        Apply an apparent-magnitude prior (``fit_catalog(prior=True)``).
        Off by default: for bright cluster members the magnitude prior is
        uninformative at best. [default: False]
    prior_file : str or None
        eazy prior table; required when ``prior`` is True.
    prior_filter : str or None
        Catalog flux column the prior magnitude is measured in, e.g.
        "f_CFHT_r"; required when ``prior`` is True.
    min_valid_bands : int
        Minimum number of valid photometric points per object; enforced in
        ``data.py`` (eazy-py ignores N_MIN_COLORS). [default: 5]
    min_snr_broadband : float
        Drop (mark missing) non-SPHEREx bands below this S/N -- genuine
        non-detections whose TEF-inflated weight can tilt the template
        blend. 0 disables. SPHEREx channels are never cut. [default: 0.0]
    spherex_prefix : str
        Band names starting with this are SPHEREx channels, built as
        per-object tophat filters from wave_um/bandwidth_um.
    extra_filters : dict
        ``{band name -> two-column curve file}`` for bands outside the
        vendored set; checked before the package registry.
    fitter : str
        eazy template solver (FITTER). [default: "nnls"]
    n_proc : int
        Worker processes for the template grid and redshift loop. [default: 4]
    extra_params : dict
        Additional eazy parameter overrides, merged last (keys are
        uppercased).
    save_zcoeffs : bool
        Persist the full (NOBJ, NZ, NTEMP) coefficient cube in arrays.npz;
        large. [default: False]
    """
    name: str = "eazy_fit"

    # Mode
    mode: str = "combo"
    z_fixed: float | None = None

    # Redshift grid
    z_min: float = 0.01
    z_max: float = 6.0
    z_step: float = 0.01
    z_step_type: str = "log"

    # Templates
    templates: str = ""
    template_pattern: str = "*_spec.dat"

    # Error model
    sys_err: float = 0.05
    tef: bool = True
    tef_file: str | None = None
    tef_scale: float = 1.0
    tef_lnp: bool = True

    # Prior
    prior: bool = False
    prior_file: str | None = None
    prior_filter: str | None = None

    # Data policy
    min_valid_bands: int = 5
    min_snr_broadband: float = 0.0
    spherex_prefix: str = SPHEREX_PREFIX
    extra_filters: dict = field(default_factory=dict)

    # Engine
    fitter: str = "nnls"
    n_proc: int = 4
    extra_params: dict = field(default_factory=dict)

    # Outputs
    save_zcoeffs: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` on an inconsistent configuration."""
        if self.mode not in ("combo", "single"):
            raise ValueError(f"mode must be 'combo' or 'single', got {self.mode!r}")
        if self.z_step_type not in ("linear", "log"):
            raise ValueError(f"z_step_type must be 'linear' or 'log', got {self.z_step_type!r}")
        if not (0.0 < self.z_min < self.z_max):
            raise ValueError(f"need 0 < z_min < z_max, got ({self.z_min}, {self.z_max})")
        if self.z_step <= 0:
            raise ValueError(f"z_step must be positive, got {self.z_step}")
        if not self.templates:
            raise ValueError(
                "config.templates is required: an eazy templates .param file, "
                "or a directory of spectra matched by template_pattern")
        if self.z_fixed is not None and not (self.z_min < self.z_fixed < self.z_max):
            raise ValueError(
                f"z_fixed={self.z_fixed} must lie strictly inside the grid "
                f"({self.z_min}, {self.z_max}); eazy silently skips edge values")
        if self.sys_err < 0 or self.tef_scale < 0:
            raise ValueError("sys_err and tef_scale must be non-negative")
        if self.min_valid_bands < 2:
            raise ValueError("min_valid_bands must be >= 2 (eazy silently skips <2-band objects)")
        if self.prior and not (self.prior_file and self.prior_filter):
            raise ValueError("prior=True requires prior_file and prior_filter")

    def to_json(self, path) -> None:
        """Write this config to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path) -> "FitConfig":
        """Load a config from a JSON file (unknown keys are ignored)."""
        with open(path) as f:
            data = json.load(f)
        known = {fld.name for fld in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path=None) -> FitConfig:
    """Return a ``FitConfig`` from a JSON file, or the defaults if ``path`` is None."""
    if path is None:
        return FitConfig()
    return FitConfig.from_json(path)
