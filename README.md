# eazy_sed_fitting

Template-fitting photometric redshifts and SEDs with the official
[eazy-py](https://github.com/gbrammer/eazy-py) pipeline (Brammer, van Dokkum,
& Coppi 2008), wrapped behind one config and one photometry CSV. The package
generates every eazy input (catalog, translate file, FILTER.RES, template
list) into a self-contained run directory, executes the official fit, and
writes compact summary products. Filter curves and a default galaxy template
atlas ship with the package, so a fit needs nothing beyond the CSV.

## Environment

Runs in the `eazy` conda env (`environment.yml`): python 3.11, eazy-py 0.8.6,
numpy, scipy, astropy, matplotlib. No sedpy and no pandas at runtime; filter
curves are vendored (see below). Invoke from the directory that contains the
package (or put it on `PYTHONPATH`):

```
conda run -n eazy python -m eazy_sed_fitting fit ...
```

## Photometry input

CSV with one row per band (or per object x band with an `id` column):

| column | meaning |
|---|---|
| `band` | e.g. `CFHT_u`, `JPLUS_J0410`, `Legacy_g`, `WISE_W1`, `SPHEREx_000` |
| `flux_uJy` | flux density in microJansky |
| `flux_err_uJy` | 1-sigma error in microJansky |
| `wave_um`, `bandwidth_um` | SPHEREx rows only: channel center and full width |

Broadband names resolve against the vendored curve set
(`python -m eazy_sed_fitting filters` lists them; `config.extra_filters` adds
your own). SPHEREx channels become per-object rectangular tophat filters.
There is no required band list, but each object must clear
`min_valid_bands` (default 5) after the data policy — eazy-py itself ignores
`N_MIN_COLORS` and silently skips under-constrained objects, so the wrapper
is the real gate.

## Quick start

```bash
python -m eazy_sed_fitting fit --phot-csv sed_input.csv \
    --z-min 0.05 --z-max 0.16 --z-step 0.001 --z-step-type linear \
    --name target1 --output-dir runs/target1 --plots --z-ref 0.106
```

```python
from eazy_sed_fitting import FitConfig, run_fit, summarize

cfg = FitConfig(name="target1",
                z_min=0.05, z_max=0.16, z_step=0.001, z_step_type="linear",
                z_fixed=0.106)
result = run_fit(cfg, "sed_input.csv", run_dir="runs/target1")
print(summarize(result))
```

Key `FitConfig` fields (defaults in parentheses): `mode` (`"combo"`; `"single"`
adds official per-template fits via `fit_single_templates`), `z_fixed` (None;
also evaluates the best-fit SED at a chosen redshift through
`fit_at_zbest`), `sys_err` (0.05 fractional error floor), `tef` (True;
the classic `TEMPLATE_ERROR.eazy_v1.0` curve at `tef_scale`=1.0),
`min_snr_broadband` (0 = off; marks broadband non-detections missing, never
SPHEREx), `prior` (False). Serialize per-target configs with
`cfg.to_json(path)` and load them with `--config`.

Templates: the packaged default is the
[Brown et al. (2014)](https://doi.org/10.1088/0067-0049/212/2/18) atlas of
129 galaxy spectra, bundled under `data/templates/brown14/` (each file
carries its original attribution header). To use an alternative set, pass
`templates=` an existing eazy `.param` file, or a directory of spectra
(two-column ASCII wavelength/f_lambda) matched by `template_pattern`.

## Quick engine (`--quick`)

eazy-py's template grid — every template integrated through every filter at
every grid redshift — takes tens of minutes to build for a many-channel
catalog against a large atlas, and per-object SPHEREx tophats mean the grid
cannot be reused between objects. `--quick` (or `run_quick_fit`, same
signature as `run_fit`) swaps in a vectorized reimplementation of the same
likelihood that builds the grid in seconds and never imports eazy-py:

```bash
python -m eazy_sed_fitting fit --phot-csv sed_input.csv --quick \
    --z-min 0.05 --z-max 0.16 --z-step 0.001 --z-step-type linear \
    --name target1 --output-dir runs/target1 --plots
```

Inputs, run-directory products, `summary.csv`/`arrays.npz` schemas,
`load_run`, and the figures are identical to the official path (the quick
run directory just omits the eazy-only input files and adds `engine.info`).
The eazy conventions it reproduces — the SYS_ERR floor, the rest-shifted
and clipped TEF with its ln P(z) normalization term, NNLS with eazy's
internal template renormalization, the 3-point parabola z_ml, single-mode
analytic amplitudes, fixed-z evaluation — are itemized in
`quick_fitting.py`'s docstring, along with its limits: no IGM absorption
(don't use it at z ≳ 2 with blue bands; a warning fires), no priors,
`nnls` only. Validated against the official engine on identical inputs:
z_ml and the P(z) percentiles agree to ~1e-5, chi2(z) to a few parts in
1e4, with identical active-template sets.

## Outputs (run directory)

```
config.json  catalog.csv  zphot.translate  FILTER.RES(.info)
templates.param  template_error.dat  zphot.param.echo
summary.csv  [singles.csv]  arrays.npz  sed_<id>.png  zscan_<id>.png
```

`summary.csv` reports three redshift estimators per object — do not mix them:

- `z_ml`: eazy's headline photo-z, the maximum of ln P(z) = -chi2/2 + tef_lnp
  (parabola-refined); -1 flags a failed/edge solution.
- `z_chi2`: the raw grid argmin of chi2(z).
- `z500` (with `z025`/`z160`/`z840`/`z975`): the P(z) median.

`arrays.npz` carries the z grid, chi2(z), ln P(z), best-fit coefficients,
model photometry, and the best-fit SED curves; `results.load_run` rehydrates
a run for plotting without re-fitting (and without eazy installed).

## Design notes and caveats

- **`standard_output` is bypassed.** eazy's output writer hardcodes
  rest-frame filter indices (UBVJ, absolute magnitudes) that index into this
  package's run-local FILTER.RES incorrectly, and its stellar-population
  columns are meaningless for shape-normalized template atlases. The package
  writes `summary.csv`/`arrays.npz` instead.
- **Template coefficients are not masses.** Atlas spectra (e.g. Brown+14)
  carry arbitrary normalizations; only the redshift, chi2, and template
  identities are physical here. Stellar-population parameters belong in a
  forward-modeling fitter (Prospector).
- **SSP caveat.** eazy-py 0.8.6 discards the age column of a templates file,
  so SSP sets fit as plain template lists with no age-vs-universe cut.
- **64-bit arrays are pinned** (`ARRAY_NBITS=64`). With eazy's float32
  default, `fit_single_templates` underflows for atlas-scale template
  fluxes: internal f_nu values ~1e-24 square to ~1e-48, which flushes to
  zero in float32, so every single-template amplitude becomes 0/0 = NaN.
  Combo fits are unaffected (`template_lsq` renormalizes each template
  before squaring), which is why the bug only surfaces in `mode="single"`.
- **Fixed-redshift fits** always go through `fit_at_zbest(zbest=...)`;
  `FIX_ZSPEC` stays off so a `z_spec` column can never silently hijack a
  photo-z run. The fixed redshift must lie strictly inside the grid.
- **Filter vendoring.** `data/filters/*.dat` are frozen transmission curves
  (provenance in each header), sourced from the sedpy registry with an
  SVO Filter Profile Service fallback for bands stock sedpy lacks (the
  J-PLUS set). Regenerate with `python vendor_filters.py` in any
  environment with numpy (sedpy optional; network needed for SVO bands).
- **Priors** are off by default and minimally supported (`prior_file` +
  `prior_filter` = a catalog flux column name).
- **Percentiles are computed by this package**, not by
  `PhotoZ.pz_percentiles`: that routine resamples ln P(z) with an Akima
  spline onto a log-spaced zoom grid, and when the zoom start lands one
  float ULP outside the fit grid (a Z_MIN-dependent accident) the leading
  NaN collapses every percentile to the grid start. `percentiles_from_lnp`
  integrates eazy's own ln P(z) directly on the fit grid.
- **macOS multiprocessing.** eazy parallelizes with multiprocessing, and
  macOS uses the spawn start method: any script that calls `run_fit` at
  import time hangs the worker pool. Keep the standard
  `if __name__ == "__main__":` guard in driver scripts (or set
  `n_proc=0` for a serial run).

## License

MIT — see [LICENSE](LICENSE).
