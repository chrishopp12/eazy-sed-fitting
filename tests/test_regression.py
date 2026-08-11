"""
test_regression.py

Regressions for Fixed Defects
---------------------------------------------------------

Five checks, one per defect that has actually bitten this package. They
are not a test suite for the engine (its numerics are validated against
the official eazy-py path elsewhere); each pins behavior that was once
wrong and must not silently return.

Run from the package directory:
    conda run -n eazy python -m pytest tests/ -q

Requirements:
  - numpy, astropy, pytest (no eazy-py, no network)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

# The package is imported as ``eazy_sed_fitting`` from its parent directory.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(IMPORT_ROOT))

from eazy_sed_fitting.config import BASE_EAZY_PARAMS, FitConfig  # noqa: E402
from eazy_sed_fitting.data import prepare_photometry, valid_rows  # noqa: E402
from eazy_sed_fitting.templates import (  # noqa: E402
    packaged_template_sets,
    prepare_templates_param,
    resolve_spectra,
    resolve_template_source,
)

ADOPTED_SET = "brown14_vac_cosmos160"


def test_omitted_templates_is_an_error() -> None:
    """A config that names no template set must not fit anything.

    The basis is the largest systematic a photometric redshift carries,
    so it may not arrive by default.
    """
    config = FitConfig(name="no_templates", templates="")
    with pytest.raises(ValueError) as excinfo:
        config.validate()
    message = str(excinfo.value)
    assert "templates is required" in message
    for name in packaged_template_sets():
        assert name in message


def test_packaged_vacuum_set_resolves_by_name(tmp_path) -> None:
    """A bare set name resolves to packaged data, vacuum sets included."""
    sets = packaged_template_sets()
    assert {"brown14", "brown14_vac", ADOPTED_SET} <= set(sets)

    source = resolve_template_source(ADOPTED_SET)
    assert source.is_dir()
    assert source.parent == PACKAGE_ROOT / "data" / "templates"

    config = FitConfig(name="adopted", templates=ADOPTED_SET,
                       template_pattern="*.dat")
    param_file = prepare_templates_param(config, tmp_path)
    spectra = [Path(line.split()[1])
               for line in param_file.read_text().splitlines() if line.strip()]
    assert len(spectra) == 160
    assert all(path.is_absolute() and path.is_file() for path in spectra)

    # The vacuum sets must be the corrected spectra, not a second copy of
    # the air atlas under a new name.
    air = resolve_template_source("brown14") / "NGC_4552_spec.dat"
    vac = resolve_template_source("brown14_vac") / "NGC_4552_spec.dat"
    assert air.is_file() and vac.is_file()
    assert air.read_bytes() != vac.read_bytes()


def test_a_partial_template_selection_says_so(capsys) -> None:
    """Fitting part of a set must not look like fitting the set.

    The default ``*_spec.dat`` takes 129 of the 160 spectra in the
    adopted set, and the basis is a factor-2.2 systematic in the fitted
    redshift -- a silent partial selection is a silently different answer.
    """
    directory = resolve_template_source(ADOPTED_SET)

    spectra = resolve_spectra(directory)
    warning = capsys.readouterr().out
    assert len(spectra) == 129
    assert "129 of 160" in warning
    assert '"*.dat"' in warning

    spectra = resolve_spectra(directory, pattern="*.dat")
    assert len(spectra) == 160
    assert capsys.readouterr().out == ""


def test_negative_detection_is_kept() -> None:
    """A real negative measurement is data, not a missing value.

    -247.22 +/- 53.75 uJy is a 4.6-sigma detection. eazy's default
    NOT_OBS_THRESHOLD of -90 (catalog units, so microJansky here) sits
    inside the range of real negative fluxes and discarded it.
    """
    flux, err = -247.22, 53.75
    phot = Table({
        "band": ["CFHT_g", "CFHT_r", "Legacy_z", "WISE_W1", "SPHEREx_042"],
        "flux_uJy": [500.0, 1000.0, 1500.0, 800.0, flux],
        "flux_err_uJy": [15.0, 20.0, 30.0, 40.0, err],
        "wave_um": [np.nan, np.nan, np.nan, np.nan, 1.5],
        "bandwidth_um": [np.nan, np.nan, np.nan, np.nan, 0.02],
    })
    config = FitConfig(name="negative", templates=ADOPTED_SET,
                       min_valid_bands=5)

    out = prepare_photometry(phot, config=config)

    assert valid_rows(out).all()
    kept = out[np.asarray(out["band"]) == "SPHEREx_042"]
    assert float(kept["flux_uJy"][0]) == flux
    assert float(kept["flux_err_uJy"][0]) == err
    # eazy must be told the same thing: the flux it treats as unobserved
    # has to lie below any real measurement.
    assert BASE_EAZY_PARAMS["NOT_OBS_THRESHOLD"] < flux


def test_import_leaves_the_matplotlib_backend_alone() -> None:
    """Importing a library must not reconfigure a caller's plotting."""
    script = (
        "import matplotlib\n"
        "matplotlib.use('template')\n"
        "import eazy_sed_fitting\n"
        "print(matplotlib.get_backend())\n"
    )
    proc = subprocess.run([sys.executable, "-c", script],
                          cwd=str(IMPORT_ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().lower() == "template"
