#!/usr/bin/env python3
"""
vendor_filters.py

Filter-Curve Vendoring (one-time maintenance)
---------------------------------------------------------

Regenerates the packaged transmission curves under ``data/filters/``,
one two-column file per band, named by the photometry-CSV band name
(e.g. ``CFHT_u.dat``). At run time the package reads only these vendored
files, so fitting requires neither sedpy nor network access; this script
is needed only when adding bands or refreshing curves.

Each band is resolved from the first source that provides it:

  1. the local sedpy registry (``BUILTIN_BAND_MAP``), when sedpy is
     installed and carries the filter;
  2. the SVO Filter Profile Service (``SVO_IDS``), fetched over HTTP --
     this covers bands absent from stock sedpy (the J-PLUS set).

Any environment with numpy works; sedpy is optional, and network access
is needed only for SVO-sourced bands. Each output file's header records
the source and date.

Requirements:
  - numpy; astro-sedpy [optional]; network access for SVO-sourced bands

Usage:
  python vendor_filters.py [--bands BAND [BAND ...]]

Examples:
  Regenerate the full packaged set:
    python vendor_filters.py
  Refresh only the J-PLUS curves:
    python vendor_filters.py --bands JPLUS_uJAVA JPLUS_J0378

Notes:
  - Curves are stored exactly as provided by the source (wavelength in
    Angstrom, dimensionless throughput); EAZY is insensitive to the
    throughput normalization.
  - SPHEREx channels are not vendored: they are per-object tophats built
    at run time from the CSV's wave_um/bandwidth_um columns.
"""

from __future__ import annotations

import argparse
import datetime
import urllib.request
from pathlib import Path

import numpy as np

FILTER_DATA_DIR = Path(__file__).resolve().parent / "data" / "filters"

SVO_URL = ("http://svo2.cab.inta-csic.es/theory/fps/getdata.php"
           "?format=ascii&id={svo_id}")

# CSV band name -> sedpy filter name. Legacy g/r are BASS and Legacy z is
# MzLS (the Legacy Surveys DR9/DR10 north-footprint instruments).
BUILTIN_BAND_MAP = {
    # Legacy Surveys (north: BASS g/r + MzLS z)
    "Legacy_g": "bass_g",
    "Legacy_r": "bass_r",
    "Legacy_z": "mzls_z",
    # SDSS
    "SDSS_u": "sdss_u0",
    "SDSS_g": "sdss_g0",
    "SDSS_r": "sdss_r0",
    "SDSS_i": "sdss_i0",
    "SDSS_z": "sdss_z0",
    # CFHT MegaCam
    "CFHT_u": "cfht_megacam_us_9301",
    "CFHT_g": "cfht_megacam_gs_9401",
    "CFHT_r": "cfht_megacam_rs_9601",
    "CFHT_i": "cfht_megacam_is_9701",
    "CFHT_z": "cfht_megacam_zs_9801",
    # Pan-STARRS1
    "PS1_g": "panstarrs_g",
    "PS1_r": "panstarrs_r",
    "PS1_i": "panstarrs_i",
    "PS1_z": "panstarrs_z",
    "PS1_y": "panstarrs_y",
    # GALEX
    "GALEX_FUV": "galex_FUV",
    "GALEX_NUV": "galex_NUV",
    # WISE (curves also apply to unWISE W1/W2 photometry)
    "WISE_W1": "wise_w1",
    "WISE_W2": "wise_w2",
    "WISE_W3": "wise_w3",
    "WISE_W4": "wise_w4",
}

# CSV band name -> SVO Filter Profile Service identifier. Used when the
# band is not resolvable through the local sedpy (stock sedpy does not
# ship the J-PLUS curves).
SVO_IDS = {
    "JPLUS_uJAVA": "OAJ/JPLUS.uJAVA",
    "JPLUS_J0378": "OAJ/JPLUS.J0378",
    "JPLUS_J0395": "OAJ/JPLUS.J0395",
    "JPLUS_J0410": "OAJ/JPLUS.J0410",
    "JPLUS_J0430": "OAJ/JPLUS.J0430",
    "JPLUS_gSDSS": "OAJ/JPLUS.gSDSS",
    "JPLUS_J0515": "OAJ/JPLUS.J0515",
    "JPLUS_rSDSS": "OAJ/JPLUS.rSDSS",
    "JPLUS_J0660": "OAJ/JPLUS.J0660",
    "JPLUS_iSDSS": "OAJ/JPLUS.iSDSS",
    "JPLUS_J0861": "OAJ/JPLUS.J0861",
    "JPLUS_zSDSS": "OAJ/JPLUS.zSDSS",
}

ALL_BANDS = list(BUILTIN_BAND_MAP) + list(SVO_IDS)


def _from_sedpy(sedpy_name: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Curve from the local sedpy registry; raises if unavailable."""
    import sedpy
    from sedpy.observate import Filter

    filt = Filter(sedpy_name)
    wave = np.asarray(filt.wavelength, float)
    thru = np.asarray(filt.transmission, float)
    return wave, thru, f"sedpy '{sedpy_name}' (astro-sedpy {sedpy.__version__})"


def _from_svo(svo_id: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Curve from the SVO Filter Profile Service; raises if unavailable."""
    with urllib.request.urlopen(SVO_URL.format(svo_id=svo_id), timeout=60) as response:
        text = response.read().decode()
    rows = [line.split() for line in text.splitlines()
            if line.strip() and not line.startswith("#")]
    if len(rows) < 2:
        raise ValueError(f"SVO returned no data for {svo_id!r}")
    curve = np.array(rows, float)
    return curve[:, 0], curve[:, 1], f"SVO FPS '{svo_id}'"


def fetch_curve(band: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve one band: local sedpy first, then SVO."""
    if band in BUILTIN_BAND_MAP:
        try:
            return _from_sedpy(BUILTIN_BAND_MAP[band])
        except Exception as err:
            if band not in SVO_IDS:
                raise RuntimeError(f"{band}: sedpy lookup failed ({err}) "
                                   "and no SVO id is defined") from err
            print(f"  {band}: sedpy unavailable ({err}); falling back to SVO")
    if band in SVO_IDS:
        return _from_svo(SVO_IDS[band])
    raise ValueError(f"unknown band {band!r}; add it to BUILTIN_BAND_MAP or SVO_IDS")


def vendor(bands: list[str]) -> None:
    """Fetch and write ``data/filters/<Band>.dat`` for each band."""
    FILTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    for band in bands:
        wave, thru, source = fetch_curve(band)
        out = FILTER_DATA_DIR / f"{band}.dat"
        header = (f"{band}\n"
                  f"vendored from {source}, {today}\n"
                  f"columns: wavelength_Angstrom throughput")
        np.savetxt(out, np.column_stack([wave, thru]), fmt="%.6e", header=header)
        print(f"  {band:12s} <- {source:44s} "
              f"({len(wave)} points, {wave.min():.0f}-{wave.max():.0f} A)")
    print(f"{len(bands)} curves -> {FILTER_DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the packaged filter transmission curves.")
    parser.add_argument("--bands", nargs="+", default=None,
                        help=f"subset of bands to refresh [default: all {len(ALL_BANDS)}]")
    args = parser.parse_args()
    vendor(args.bands or ALL_BANDS)


if __name__ == "__main__":
    main()
