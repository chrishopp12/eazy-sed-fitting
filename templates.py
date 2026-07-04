"""
templates.py

Template-Set Resolution
---------------------------------------------------------

Turns ``config.templates`` into the templates ``.param`` file a PhotoZ run
consumes:

  - an empty ``config.templates`` selects the packaged default, the
    Brown et al. (2014) atlas of 129 galaxy spectra (CDS J/ApJS/212/18),
    stored under ``data/templates/brown14`` with each file carrying its
    original attribution header;
  - a path to an existing eazy ``.param`` file is used as-is (its internal
    spectrum paths must be absolute or eazy-resolvable);
  - a directory is globbed with ``config.template_pattern`` (falling back
    to ``*.dat``) and a ``.param`` file listing the spectra by absolute
    path is written into the run directory.

Spectra must be readable by ``eazy.templates.Template``: two-column ASCII
(wavelength Angstrom, f_lambda; ``#`` comments allowed -- the Brown et al.
2014 atlas files work as-is) or FITS/CSV/ECSV tables with ``wave`` and
``flux`` columns. Image-array FITS spectra (e.g. E-MILES SSPs) need
conversion first.

Notes:
  - SSP caveat: eazy-py 0.8.6 reads only the path and wavelength-scale
    columns of a templates file and discards the age column, so no
    age-vs-universe cut is applied at any redshift. SSP catalogs therefore
    fit as plain template lists; physically-motivated SSP work (ages,
    masses, SFHs) belongs in a forward-modeling fitter such as Prospector.
"""

from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_TEMPLATE_DIR, FitConfig


def resolve_spectra(directory, *, pattern: str = "*_spec.dat") -> list[Path]:
    """Sorted spectrum paths in a template directory.

    Falls back to ``*.dat`` when ``pattern`` matches nothing, so a generic
    directory of two-column spectra works without configuration.
    """
    directory = Path(directory).expanduser()
    spectra = sorted(directory.glob(pattern))
    if not spectra and pattern != "*.dat":
        spectra = sorted(directory.glob("*.dat"))
    if not spectra:
        raise ValueError(f"no template spectra matching {pattern!r} (or *.dat) in {directory}")
    return spectra


def write_templates_param(spectra: list[Path], out_path) -> Path:
    """Write an eazy templates ``.param`` file listing ``spectra``.

    Rows are ``<number> <absolute path> 1.0``; eazy-py reads only the path
    and the wavelength-to-Angstrom factor.
    """
    out_path = Path(out_path)
    lines = [f"{i} {path.resolve()} 1.0" for i, path in enumerate(spectra, start=1)]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def prepare_templates_param(config: FitConfig, run_dir) -> Path:
    """Resolve ``config.templates`` to the ``.param`` file for this run.

    Parameters
    ----------
    config : FitConfig
        ``templates`` is an existing ``.param`` file, a spectrum
        directory, or empty for the packaged Brown et al. (2014) atlas;
        ``template_pattern`` applies in directory mode.
    run_dir : Path
        Destination for a generated ``templates.param``.

    Returns
    -------
    param_path : Path
        Absolute path handed to eazy as TEMPLATES_FILE.
    """
    spec = Path(config.templates).expanduser() if config.templates else DEFAULT_TEMPLATE_DIR
    if spec.is_file() and spec.suffix == ".param":
        return spec.resolve()
    if spec.is_dir():
        spectra = resolve_spectra(spec, pattern=config.template_pattern)
        return write_templates_param(spectra, Path(run_dir) / "templates.param")
    raise ValueError(
        f"config.templates={config.templates!r} is neither a .param file nor a directory")
