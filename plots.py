"""
plots.py

Diagnostic Figures for eazy_sed_fitting Runs
---------------------------------------------------------

Two figures per object, drawn from a ``FitResult`` (live or rehydrated
with ``results.load_run`` -- no eazy import needed here):

  - ``plot_sed``: the plot_map_sed layout shared with
    ``prospector_sed_fitting`` -- a short chi-residual panel over a tall
    SED panel (best-fit template spectrum, observed photometry colored by
    instrument, model photometry), log wavelength, f_nu in microJansky.
    Residuals use the photometric error (SYS_ERR included); the TEF
    variance inflation lives in the likelihood, not the error bars.
  - ``plot_zscan``: delta-chi2 versus redshift next to the normalized
    P(z); in single mode the chi2 panel overlays the per-template curves.

Requirements:
  - numpy, matplotlib
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .config import SPHEREX_PREFIX
from .results import FitResult

# Per-instrument marker colors (shared with prospector_sed_fitting's
# plot_map_sed and sed_photoz's plot_sed_combo); SPHEREx de-emphasized.
INSTRUMENT_COLOR = {
    "GALEX": "darkviolet", "SDSS": "seagreen", "Legacy": "olivedrab",
    "CFHT": "steelblue", "JPLUS": "teal", "PS1": "darkgoldenrod",
    "WISE": "darkorange", "SPHEREx": "0.55",
}
INSTRUMENT_LABEL = {"JPLUS": "J-PLUS"}
SED_XTICKS = [1e3, 2e3, 3e3, 5e3, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5]


def instrument_of(band: str, spherex_prefix: str = SPHEREX_PREFIX) -> str:
    """Instrument label for a band ("CFHT_u" -> "CFHT"; SPHEREx channels pooled)."""
    if str(band).startswith(spherex_prefix):
        return "SPHEREx"
    return str(band).split("_")[0]


def _apply_xaxis(ax, wave) -> None:
    """Log wavelength axis with the shared tick set, in observed Angstrom."""
    lo, hi = float(np.min(wave)) * 0.8, float(np.max(wave)) * 1.25
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ticks = [t for t in SED_XTICKS if lo <= t <= hi]
    if ticks:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1e4 else f"{x / 1e3:.0f}k"))
    ax.set_xlabel(r"$\lambda_{\rm obs}$ ($\AA$)")


def _apply_yaxis(ax, flux, model, spec_wave, spec_fnu, pad=3.0) -> None:
    """Log flux limits from the photometry and the in-view model spectrum."""
    lo, hi = ax.get_xlim()
    values = [np.asarray(flux, float), np.asarray(model, float)]
    inview = (spec_wave >= lo) & (spec_wave <= hi)
    spec = spec_fnu[inview]
    spec = spec[np.isfinite(spec) & (spec > 0)]
    if spec.size:
        values.append(np.percentile(spec, [1, 99]))
    stacked = np.concatenate([np.atleast_1d(v) for v in values])
    stacked = stacked[np.isfinite(stacked) & (stacked > 0)]
    ax.set_yscale("log")
    if stacked.size:
        ax.set_ylim(stacked.min() / pad, stacked.max() * pad)


def plot_sed(result: FitResult, iobj: int = 0, *,
             fixed: bool = False,
             z_ref: float | None = None,
             save_dir=None,
             color_by_instrument: bool = True) -> Path | None:
    """SED + chi-residual figure for one object.

    Parameters
    ----------
    result : FitResult
        Live or rehydrated run.
    iobj : int
        Object index. [default: 0]
    fixed : bool
        Plot the fixed-z solution instead of the photo-z one. [default: False]
    z_ref : float or None
        Reference redshift drawn in the title. [default: None]
    save_dir : str or Path or None
        Output directory; None uses the run directory. [default: None]
    color_by_instrument : bool
        Color photometry by instrument; False plots black points. [default: True]

    Returns
    -------
    png_path : Path or None
        The written figure, or None if the object has no stored SED.
    """
    sed = (result.seds_fixed if fixed else result.seds)[iobj]
    if sed is None:
        print(f"no stored SED for object index {iobj} ({'fixed' if fixed else 'photo-z'})")
        return None

    oid = result.ids[iobj]
    valid = np.asarray(result.ok_data[iobj], bool)
    wave = np.asarray(result.pivot, float)[valid]
    fobs = np.asarray(sed["fobs"], float)[valid]
    efobs = np.asarray(sed["efobs"], float)[valid]
    model = np.asarray(sed["model"], float)[valid]
    bands = [b for b, ok in zip(result.bands, valid) if ok]
    insts = np.array([instrument_of(b, result.config.spherex_prefix) for b in bands])

    chi = (fobs - model) / efobs
    n_active = int(((result.coeffs_fixed if fixed else result.coeffs_best)[iobj] > 0).sum())
    ndof = max(1, int(result.nusefilt[iobj]) - n_active - 1)
    redchi2 = float(np.sum(chi ** 2)) / ndof

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw=dict(height_ratios=[1, 4]), sharex=True)

    ax = axes[1]
    ax.plot(sed["templz"], sed["templf"], color="firebrick", lw=0.8, alpha=0.8,
            label="Best-fit spectrum")
    if color_by_instrument:
        for inst in dict.fromkeys(insts):
            m = insts == inst
            is_sx = inst == "SPHEREx"
            ax.errorbar(wave[m], fobs[m], efobs[m], linestyle="", marker="o",
                        color=INSTRUMENT_COLOR.get(inst, "slategray"),
                        ms=3 if is_sx else 7, mec="none" if is_sx else "k", mew=0.5,
                        elinewidth=0.6, alpha=0.55 if is_sx else 0.95,
                        zorder=8 if is_sx else 10,
                        label=INSTRUMENT_LABEL.get(inst, inst))
        sx = insts == "SPHEREx"
        if (~sx).any():
            ax.plot(wave[~sx], model[~sx], linestyle="", marker="x", color="k",
                    ms=5, mew=1.0, zorder=11, label="Model phot")
        if sx.any():
            ax.plot(wave[sx], model[sx], linestyle="", marker="x", color="k",
                    ms=2, mew=0.5, alpha=0.7, zorder=9)
    else:
        ax.errorbar(wave, fobs, efobs, linestyle="", marker="o", color="k",
                    zorder=10, label="Observed photometry")
        ax.plot(wave, model, linestyle="", marker="s", markersize=10,
                mec="orange", mew=3, mfc="none", label="Model photometry")
    ax.set_ylabel(r"$f_\nu$ ($\mu$Jy)")
    _apply_xaxis(ax, wave)
    _apply_yaxis(ax, fobs, model, np.asarray(sed["templz"], float),
                 np.asarray(sed["templf"], float))
    ax.legend(fontsize=8, loc="upper right", ncol=2 if color_by_instrument else 1)
    ref = f", ref = {z_ref:.4f}" if z_ref is not None else ""
    tag = "fixed-z" if fixed else "photo-z"
    ax.set_title(f"{oid} -- eazy {tag} solution  "
                 f"(z = {sed['z']:.4f}, $\\chi^2_\\nu$ = {redchi2:.1f}{ref})")

    ax = axes[0]
    if color_by_instrument:
        for inst in dict.fromkeys(insts):
            m = insts == inst
            is_sx = inst == "SPHEREx"
            ax.plot(wave[m], chi[m], linestyle="", marker="o",
                    color=INSTRUMENT_COLOR.get(inst, "slategray"),
                    ms=3 if is_sx else 5, alpha=0.55 if is_sx else 0.95)
    else:
        ax.plot(wave, chi, linestyle="", marker="o", color="k")
    ax.axhline(0, color="k", linestyle=":")
    ax.set_ylabel(r"$\chi$")
    ax.set_ylim(-5, 5)

    plt.tight_layout()
    save_dir = Path(save_dir) if save_dir else Path(result.run_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_fixed" if fixed else ""
    png_path = save_dir / f"sed{suffix}_{oid}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


def plot_zscan(result: FitResult, iobj: int = 0, *,
               z_ref: float | None = None,
               save_dir=None,
               n_singles: int = 8) -> Path:
    """Delta-chi2 and P(z) panels for one object.

    In single mode the chi2 panel overlays the ``n_singles`` best
    per-template curves (best template highlighted).
    """
    oid = result.ids[iobj]
    zgrid = np.asarray(result.zgrid, float)
    chi2 = np.asarray(result.chi2_fit[iobj], float)
    dchi2 = chi2 - np.nanmin(chi2)

    if result.lnp is not None:
        lnp = np.asarray(result.lnp[iobj], float)
    else:
        lnp = -0.5 * dchi2
    pz = np.exp(lnp - np.nanmax(lnp))
    norm = np.trapezoid(pz, zgrid)
    if norm > 0:
        pz = pz / norm

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(zgrid, dchi2, "-", color="k", lw=1.4, label="combo", zorder=10)
    if result.singles_chi2 is not None:
        singles = np.asarray(result.singles_chi2[:, iobj, :], float)
        order = np.argsort(singles.min(axis=1))[:n_singles]
        floor = np.nanmin(chi2)
        for rank, t in enumerate(order):
            best = rank == 0
            ax.plot(zgrid, singles[t] - floor, "-",
                    color="firebrick" if best else "0.75",
                    lw=1.2 if best else 0.7, zorder=5 if best else 4,
                    label=(f"best single: {result.template_names[t]}" if best else None))
    ax.axhline(1, ls=":", color="0.6")
    ax.set_ylim(0, 30)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$\Delta\chi^2$")

    ax = axes[1]
    ax.plot(zgrid, pz, "-", color="k", lw=1.4)
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$P(z)$")

    for ax in axes:
        if result.z_ml[iobj] > 0:
            ax.axvline(result.z_ml[iobj], color="C1", lw=1.2,
                       label=f"$z_{{\\rm ml}}$ = {result.z_ml[iobj]:.4f}")
        if result.z_fixed is not None:
            ax.axvline(result.z_fixed, color="C0", lw=1.0, ls="-.",
                       label=f"$z_{{\\rm fixed}}$ = {result.z_fixed:.4f}")
        if z_ref is not None:
            ax.axvline(z_ref, ls="--", color="k", lw=1.0, label=f"ref = {z_ref:.4f}")
        ax.legend(fontsize=8)

    mode = result.config.mode
    fig.suptitle(f"{oid}: eazy redshift scan "
                 f"({len(result.template_names)} templates, {mode})")
    fig.tight_layout()

    save_dir = Path(save_dir) if save_dir else Path(result.run_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"zscan_{oid}.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def generate_plots(result: FitResult, *, z_ref: float | None = None) -> list[Path]:
    """All figures for every object in a run; returns the written paths."""
    written = []
    for iobj in range(len(result.ids)):
        for fixed in (False, True) if result.z_fixed is not None else (False,):
            path = plot_sed(result, iobj, fixed=fixed, z_ref=z_ref)
            if path:
                written.append(path)
        written.append(plot_zscan(result, iobj, z_ref=z_ref))
    return written
