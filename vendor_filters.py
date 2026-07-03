#!/usr/bin/env python3
"""
vendor_filters.py

Filter-Curve Vendoring (one-time maintenance)
---------------------------------------------------------

Extracts the built-in broadband transmission curves from sedpy and writes
them as plain two-column files under ``data/filters/``, one per band, named
by the photometry-CSV band name (e.g. ``CFHT_u.dat``). At runtime the
package reads these vendored files, so fitting requires neither sedpy nor
a band-to-filter map.

Run this only when adding bands or refreshing curves, in an environment
whose sedpy installation carries every filter in ``BUILTIN_BAND_MAP``
(the J-PLUS curves are hand-installed .par files, present in the
``prospector_c3k`` env only):

Requirements:
  - numpy, sedpy

Usage:
  conda run -n prospector_c3k python vendor_filters.py

Notes:
  - Curves are written exactly as sedpy provides them (wavelength in
    Angstrom, dimensionless throughput); EAZY is insensitive to the
    throughput normalization.
  - SPHEREx channels are not vendored: they are per-object tophats built
    at run time from the CSV's wave_um/bandwidth_um columns.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

FILTER_DATA_DIR = Path(__file__).resolve().parent / "data" / "filters"

# CSV band name -> sedpy filter name. Legacy g/r are BASS and Legacy z is
# MzLS (the DR9 north footprint instruments used for the A1925 photometry).
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
    # J-PLUS (12 bands; hand-installed SVO curves in the prospector_c3k sedpy)
    "JPLUS_uJAVA": "jplus_ujava",
    "JPLUS_J0378": "jplus_j0378",
    "JPLUS_J0395": "jplus_j0395",
    "JPLUS_J0410": "jplus_j0410",
    "JPLUS_J0430": "jplus_j0430",
    "JPLUS_gSDSS": "jplus_gsdss",
    "JPLUS_J0515": "jplus_j0515",
    "JPLUS_rSDSS": "jplus_rsdss",
    "JPLUS_J0660": "jplus_j0660",
    "JPLUS_iSDSS": "jplus_isdss",
    "JPLUS_J0861": "jplus_j0861",
    "JPLUS_zSDSS": "jplus_zsdss",
    # GALEX
    "GALEX_FUV": "galex_FUV",
    "GALEX_NUV": "galex_NUV",
    # WISE (curves also apply to unWISE W1/W2 photometry)
    "WISE_W1": "wise_w1",
    "WISE_W2": "wise_w2",
    "WISE_W3": "wise_w3",
    "WISE_W4": "wise_w4",
}


def vendor_all(band_map: dict[str, str] = BUILTIN_BAND_MAP) -> None:
    """Dump every band in ``band_map`` to ``data/filters/<Band>.dat``."""
    import sedpy
    from sedpy.observate import load_filters

    FILTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    bands = list(band_map)
    filters = load_filters([band_map[b] for b in bands])

    for band, filt in zip(bands, filters):
        wave = np.asarray(filt.wavelength, float)
        thru = np.asarray(filt.transmission, float)
        out = FILTER_DATA_DIR / f"{band}.dat"
        header = (f"{band}\n"
                  f"vendored from sedpy '{band_map[band]}' "
                  f"(astro-sedpy {sedpy.__version__}), {today}\n"
                  f"columns: wavelength_Angstrom throughput")
        np.savetxt(out, np.column_stack([wave, thru]), fmt="%.6e", header=header)
        print(f"  {band:12s} <- {band_map[band]:24s} "
              f"({len(wave)} points, {wave.min():.0f}-{wave.max():.0f} A)")

    print(f"{len(bands)} curves -> {FILTER_DATA_DIR}")


if __name__ == "__main__":
    vendor_all()
