"""
data.py

Photometry Input and Data Policy
---------------------------------------------------------

Reads the SED-input photometry CSV and applies the wrapper-side data
policy before anything reaches eazy-py:

  - non-finite fluxes and non-positive errors become missing values
    (``MISSING_FLUX`` = -99, below eazy's NOT_OBS_THRESHOLD);
  - an optional broadband S/N cut marks non-detections missing (SPHEREx
    channels are never cut);
  - a minimum-valid-band count is enforced with a hard error, because
    eazy-py ignores N_MIN_COLORS and silently skips objects with fewer
    than two valid bands.

CSV schema (one row per band, or per object x band with an ``id`` column):
  band            e.g. CFHT_u, JPLUS_J0410, Legacy_g, WISE_W1, SPHEREx_000
  flux_uJy        flux density in microJansky
  flux_err_uJy    1-sigma error in microJansky
  wave_um         SPHEREx rows only: channel center (micron)
  bandwidth_um    SPHEREx rows only: channel full width (micron)

Multiple objects must share an identical band sequence (the EAZY catalog
model: one filter set per catalog). Per-object SPHEREx channel wavelengths
that differ between objects are rejected -- run those objects separately.

Requirements:
  - numpy, astropy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from .config import FitConfig, MISSING_FLUX, SPHEREX_PREFIX

REQUIRED_COLUMNS = ("band", "flux_uJy", "flux_err_uJy")
OPTIONAL_FLOAT_COLUMNS = ("wave_um", "bandwidth_um")


def is_spherex(band, prefix: str = SPHEREX_PREFIX) -> bool:
    """True if ``band`` names a SPHEREx channel (per-object tophat filter)."""
    return str(band).startswith(prefix)


def valid_rows(table: Table) -> np.ndarray:
    """Boolean mask of rows carrying a usable measurement.

    A row is valid when its flux is finite and above eazy's missing-data
    region and its error is positive -- the same criterion eazy applies.
    """
    flux = np.asarray(table["flux_uJy"], float)
    err = np.asarray(table["flux_err_uJy"], float)
    return np.isfinite(flux) & (flux > -90.0) & np.isfinite(err) & (err > 0)


def object_ids(phot: Table) -> list[str]:
    """Object ids in first-appearance order."""
    seen: dict[str, None] = {}
    for oid in phot["id"]:
        seen.setdefault(str(oid), None)
    return list(seen)


def _float_array(table: Table, name: str) -> np.ndarray:
    """Column as a float array; masked or absent entries become NaN."""
    if name not in table.colnames:
        return np.full(len(table), np.nan)
    col = table[name]
    if hasattr(col, "filled"):
        col = col.filled(np.nan)
    return np.asarray(col, float)


def _normalize_table(raw: Table) -> Table:
    """Coerce an input table to the canonical column set and dtypes."""
    for col in REQUIRED_COLUMNS:
        if col not in raw.colnames:
            raise ValueError(f"photometry table missing required column {col!r}")
    phot = Table()
    if "id" in raw.colnames:
        phot["id"] = np.asarray(raw["id"]).astype(str)
    else:
        phot["id"] = np.full(len(raw), "", dtype=object)
    phot["band"] = np.asarray(raw["band"]).astype(str)
    phot["flux_uJy"] = _float_array(raw, "flux_uJy")
    phot["flux_err_uJy"] = _float_array(raw, "flux_err_uJy")
    for col in OPTIONAL_FLOAT_COLUMNS:
        phot[col] = _float_array(raw, col)
    return phot


def read_photometry(csv_path) -> Table:
    """Read a SED-input CSV into the normalized photometry table.

    Parameters
    ----------
    csv_path : str or Path
        CSV with the schema in the module header.

    Returns
    -------
    phot : Table
        Columns ``id, band, flux_uJy, flux_err_uJy, wave_um, bandwidth_um``
        (``id`` filled with "" when the CSV has none; ``prepare_photometry``
        substitutes the config name).
    """
    raw = Table.read(str(csv_path), format="ascii.csv")
    try:
        return _normalize_table(raw)
    except ValueError as err:
        raise ValueError(f"{csv_path}: {err}") from None


def apply_data_policy(phot: Table, *, config: FitConfig) -> Table:
    """Return a copy of ``phot`` with the data policy applied.

    Marks unusable rows missing (never drops them, so every object keeps
    the same band sequence), applies the broadband S/N cut, and enforces
    ``config.min_valid_bands`` per object.
    """
    out = phot.copy()
    flux = np.asarray(out["flux_uJy"], float)
    err = np.asarray(out["flux_err_uJy"], float)
    sx = np.array([is_spherex(b, config.spherex_prefix) for b in out["band"]])

    # Unusable measurements -> missing (eazy treats fnu < NOT_OBS_THRESHOLD
    # or efnu <= 0 as not observed; -99/-99 makes that explicit and stable).
    bad = ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0)

    # Broadband non-detections -> missing. With the TEF inflating SPHEREx
    # errors, a single S/N ~ 1 broadband point can tilt the template blend
    # (a phantom pull, not information). SPHEREx channels are never cut:
    # band count is the intended signal of the many-narrow-band design.
    if config.min_snr_broadband > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            low = ~sx & ~bad & (flux / err < config.min_snr_broadband)
        n_low = int(low.sum())
        if n_low:
            names = ", ".join(str(b) for b in out["band"][low])
            print(f"data policy: {n_low} broadband band(s) below "
                  f"S/N {config.min_snr_broadband:g} marked missing: {names}")
        bad |= low

    out["flux_uJy"][bad] = MISSING_FLUX
    out["flux_err_uJy"][bad] = MISSING_FLUX

    # Minimum valid bands per object -- the wrapper is the only real gate
    # (eazy-py ignores N_MIN_COLORS and silently skips <2-band objects).
    ok = valid_rows(out)
    for oid in object_ids(out):
        n_valid = int(ok[np.asarray(out["id"]) == oid].sum())
        if n_valid < config.min_valid_bands:
            raise ValueError(
                f"object {oid or config.name!r}: {n_valid} valid bands, "
                f"below min_valid_bands={config.min_valid_bands}")
    return out


def prepare_photometry(phot, *, config: FitConfig) -> Table:
    """Normalize, policy-filter, and validate photometry for one fit.

    Parameters
    ----------
    phot : str, Path, or Table
        SED-input CSV path, or an equivalent in-memory table.
    config : FitConfig
        Supplies the data policy and the fallback object id.

    Returns
    -------
    phot : Table
        Policy-applied table ready for ``fitting.run_fit``.
    """
    if isinstance(phot, (str, Path)):
        phot = read_photometry(phot)
    else:
        phot = _normalize_table(phot)
    if not any(str(v) for v in phot["id"]):
        phot["id"] = np.full(len(phot), config.name, dtype=object)
    phot = apply_data_policy(phot, config=config)
    _check_band_consistency(phot, config=config)
    return phot


def load_photometry(csv_path, *, config: FitConfig) -> Table:
    """Read and prepare a photometry CSV (see ``prepare_photometry``)."""
    return prepare_photometry(csv_path, config=config)


def band_metadata(phot: Table) -> Table:
    """Per-band metadata in fit order (the first object's band sequence).

    Returns a table with ``band, wave_um, bandwidth_um`` -- the input the
    filter builder needs to write FILTER.RES.
    """
    first = np.asarray(phot["id"]) == object_ids(phot)[0]
    return phot[first]["band", "wave_um", "bandwidth_um"].copy()


# ------------------------------------
# Internal checks
# ------------------------------------

def _check_band_consistency(phot: Table, *, config: FitConfig) -> None:
    """All objects must share one band sequence (one filter set per catalog)."""
    ids = object_ids(phot)
    ref = phot[np.asarray(phot["id"]) == ids[0]]
    ref_bands = list(ref["band"])
    for oid in ids[1:]:
        sub = phot[np.asarray(phot["id"]) == oid]
        if list(sub["band"]) != ref_bands:
            raise ValueError(
                f"object {oid!r} has a different band sequence than {ids[0]!r}; "
                "an EAZY catalog carries one filter set -- fit such objects separately")
        for col in OPTIONAL_FLOAT_COLUMNS:
            a = np.asarray(ref[col], float)
            b = np.asarray(sub[col], float)
            both = np.isfinite(a) & np.isfinite(b)
            if not np.allclose(a[both], b[both], rtol=1e-6, atol=0):
                raise ValueError(
                    f"object {oid!r}: SPHEREx channel {col} differs from {ids[0]!r}; "
                    "per-object channels imply one object per fit")
            if np.any(np.isfinite(a) != np.isfinite(b)):
                raise ValueError(
                    f"object {oid!r}: SPHEREx metadata pattern differs from {ids[0]!r}")

    # SPHEREx rows need their tophat geometry.
    sx = np.array([is_spherex(b, config.spherex_prefix) for b in phot["band"]])
    if sx.any():
        wave = np.asarray(phot["wave_um"], float)
        width = np.asarray(phot["bandwidth_um"], float)
        incomplete = sx & (~np.isfinite(wave) | ~np.isfinite(width) | (width <= 0))
        if incomplete.any():
            names = ", ".join(sorted(set(map(str, phot["band"][incomplete]))))
            raise ValueError(f"SPHEREx rows missing wave_um/bandwidth_um: {names}")
