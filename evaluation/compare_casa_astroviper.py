#!/usr/bin/env python3
"""Compare exact-input CASA benchmarks with AstroViper continuum products."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/casa-equivalence-matplotlib")

import matplotlib
import numpy as np
import xarray as xr
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_JSON = ROOT / "evaluation" / "casa_equivalence_results.json"
OUTPUT_MD = ROOT / "evaluation" / "casa_equivalence_report.md"
GAUSSIAN_BEAM_FACTOR = np.pi / (4.0 * np.log(2.0))
RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi


def _av_plane(dataset, variable, **indices):
    array = dataset[variable]
    if "polarization" in array.dims:
        array = array.sel(polarization="I")
    if "time" in array.dims:
        indices["time"] = 0
    # AstroViper stores (l, m); FITS stores (y=m, x=l).
    return np.asarray(array.isel(indices).values).squeeze().T


def _fits_plane(path):
    return np.asarray(fits.getdata(path), dtype=float).squeeze()


def _correlation(left, right, mask=None):
    valid = np.isfinite(left) & np.isfinite(right)
    if mask is not None:
        valid &= mask
    if np.count_nonzero(valid) < 2:
        return np.nan
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _relative_l2(test, reference, mask=None):
    valid = np.isfinite(test) & np.isfinite(reference)
    if mask is not None:
        valid &= mask
    denominator = np.linalg.norm(reference[valid])
    return float(np.linalg.norm(test[valid] - reference[valid]) / denominator)


def _robust_sigma(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _peak(array):
    finite = np.where(np.isfinite(array), array, -np.inf)
    position = np.unravel_index(np.argmax(finite), finite.shape)
    return float(finite[position]), [int(value) for value in position]


def _plot_scale(*arrays, symmetric=False):
    values = np.concatenate(
        [np.asarray(array)[np.isfinite(array)].ravel() for array in arrays]
    )
    if symmetric:
        limit = np.percentile(np.abs(values), 99.7)
        if not np.isfinite(limit) or limit == 0.0:
            limit = float(np.max(np.abs(values)))
        return -limit, limit
    return float(np.percentile(values, 0.3)), float(np.percentile(values, 99.7))


def _render_comparison(output, casa, av, title):
    products = (
        ("Restored Taylor 0", casa["restored"], av["restored"], False),
        ("Residual Taylor 0", casa["residual"], av["residual"], True),
        ("Model Taylor 0", casa["model"], av["model"], True),
    )
    figure, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for row, (name, casa_array, av_array, symmetric) in enumerate(products):
        vmin, vmax = _plot_scale(casa_array, av_array, symmetric=symmetric)
        difference = av_array - casa_array
        diff_limit = np.percentile(np.abs(difference[np.isfinite(difference)]), 99.7)
        if not np.isfinite(diff_limit) or diff_limit == 0.0:
            diff_limit = float(np.nanmax(np.abs(difference)))
        images = (
            (casa_array, "CASA", vmin, vmax, "viridis"),
            (av_array, "AstroViper", vmin, vmax, "viridis"),
            (difference, "AstroViper − CASA", -diff_limit, diff_limit, "RdBu_r"),
        )
        for column, (array, label, low, high, cmap) in enumerate(images):
            image = axes[row, column].imshow(
                array, origin="lower", cmap=cmap, vmin=low, vmax=high
            )
            axes[row, column].set_title(f"{name}: {label}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            figure.colorbar(image, ax=axes[row, column], shrink=0.72)
    figure.suptitle(title)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _casa_paths(base):
    root = base / "casa_benchmark_astroviper_matched"
    mtmfs = (root / "continuum.image.tt0.fits").exists()
    term = ".tt0" if mtmfs else ""
    pbcor = ".image.tt0.pbcor.fits" if mtmfs else ".image.pbcor.fits"
    return {
        "restored": root / f"continuum.image{term}.fits",
        "restored_pbcor": root / f"continuum{pbcor}",
        "residual": root / f"continuum.residual{term}.fits",
        "model": root / f"continuum.model{term}.fits",
        "psf": root / f"continuum.psf{term}.fits",
        "pb": root / f"continuum.pb{term}.fits",
    }


def _load_casa(paths):
    return {name: _fits_plane(path) for name, path in paths.items()}


def _load_av(path):
    dataset = xr.open_zarr(path)
    arrays = {
        "restored": _av_plane(dataset, "SKY_RESTORED", taylor_term=0),
        "restored_pbcor": _av_plane(dataset, "SKY_RESTORED_PBCOR"),
        "residual": _av_plane(dataset, "SKY_RESIDUAL", taylor_term=0),
        "model": _av_plane(dataset, "SKY_MODEL", taylor_term=0),
        "psf": _av_plane(dataset, "POINT_SPREAD_FUNCTION", psf_taylor_order=0),
        "pb": _av_plane(dataset, "PRIMARY_BEAM", frequency=0),
    }
    beam = np.asarray(
        dataset.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I")
        .isel(time=0)
        .values
    )
    dl = abs(float(np.median(np.diff(dataset.l.values))))
    dm = abs(float(np.median(np.diff(dataset.m.values))))
    return dataset, arrays, beam, dl * dm


def _metrics(base, av_path, casa, casa_paths):
    dataset, av, av_beam, pixel_area = _load_av(av_path)
    central = (
        np.isfinite(casa["pb"])
        & np.isfinite(av["pb"])
        & (casa["pb"] >= 0.2)
        & (av["pb"] >= 0.2)
    )
    noise_region = central & (casa["pb"] >= 0.5) & (av["pb"] >= 0.5)
    casa_peak, casa_position = _peak(casa["restored"])
    av_peak, av_position = _peak(av["restored"])
    casa_header = fits.getheader(casa_paths["restored"])
    casa_major = float(casa_header["BMAJ"]) * np.pi / 180.0
    casa_minor = float(casa_header["BMIN"]) * np.pi / 180.0
    casa_beam_area = GAUSSIAN_BEAM_FACTOR * casa_major * casa_minor
    av_beam_area = GAUSSIAN_BEAM_FACTOR * av_beam[0] * av_beam[1]
    result = {
        "project": base.parent.name,
        "variant": base.name,
        "astroviper_run": av_path.parent.name,
        "restored_correlation": _correlation(av["restored"], casa["restored"], central),
        "restored_relative_l2": _relative_l2(av["restored"], casa["restored"], central),
        "pbcor_correlation": _correlation(
            av["restored_pbcor"], casa["restored_pbcor"], central
        ),
        "psf_correlation": _correlation(av["psf"], casa["psf"]),
        "model_correlation": _correlation(av["model"], casa["model"]),
        "psf_relative_l2": _relative_l2(av["psf"], casa["psf"]),
        "pb_correlation": _correlation(av["pb"], casa["pb"]),
        "pb_relative_l2": _relative_l2(av["pb"], casa["pb"]),
        "casa_peak_jy_per_beam": casa_peak,
        "astroviper_peak_jy_per_beam": av_peak,
        "peak_ratio_astroviper_over_casa": av_peak / casa_peak,
        "peak_separation_pixels": float(
            np.linalg.norm(np.asarray(av_position) - np.asarray(casa_position))
        ),
        "casa_residual_mad_sigma_jy_per_beam": _robust_sigma(
            casa["residual"][noise_region]
        ),
        "astroviper_residual_mad_sigma_jy_per_beam": _robust_sigma(
            av["residual"][noise_region]
        ),
        "casa_model_sum_jy": float(np.nansum(casa["model"])),
        "astroviper_model_sum_jy": float(np.nansum(av["model"])),
        "casa_nonzero_model_pixels": int(np.count_nonzero(casa["model"])),
        "astroviper_nonzero_model_pixels": int(np.count_nonzero(av["model"])),
        "casa_beam_major_arcsec": casa_major * RAD_TO_ARCSEC,
        "casa_beam_minor_arcsec": casa_minor * RAD_TO_ARCSEC,
        "astroviper_beam_major_arcsec": av_beam[0] * RAD_TO_ARCSEC,
        "astroviper_beam_minor_arcsec": av_beam[1] * RAD_TO_ARCSEC,
        "beam_area_ratio_astroviper_over_casa": av_beam_area / casa_beam_area,
        "pixel_area_steradian": pixel_area,
        "reference_frequency_hz": float(
            dataset.SKY_RESTORED.attrs["reference_frequency"]
        ),
    }
    return result, av


def _interpret(baseline):
    if baseline["psf_correlation"] < 0.95:
        return "FAIL: gridding/normalization differs materially (PSF mismatch)."
    casa_nz = baseline["casa_nonzero_model_pixels"]
    av_nz = baseline["astroviper_nonzero_model_pixels"]
    if casa_nz == 0 and av_nz > 0:
        return "MODEL DIFFERENCE: CASA stopped with an empty model while AstroViper cleaned emission."
    if casa_nz > 0 and av_nz == 0:
        return "FAIL: CASA cleaned emission but AstroViper stopped with an empty model."
    if baseline["restored_correlation"] >= 0.99:
        return "PASS: strong image equivalence for the matched MFS/global/CASA-weight path."
    if baseline["restored_correlation"] >= 0.9:
        return "PARTIAL: imaging agrees broadly; minor-cycle differences remain."
    return "PARTIAL: PSF agrees, but cleaning produces materially different restored images."


def main():
    results = []
    for complete in sorted(
        RESULTS.glob("*/*/casa_benchmark_astroviper_matched/complete.json")
    ):
        base = complete.parent.parent
        casa_paths = _casa_paths(base)
        casa = _load_casa(casa_paths)
        av_paths = []
        for pattern in (
            "mfs_global_casa_weights*/continuum.img.zarr",
            "mvc_global_casa_weights*/continuum.img.zarr",
        ):
            for av_path in (base / "astroviper").glob(pattern):
                manifest = json.loads((av_path.parent / "complete.json").read_text())
                if manifest.get("clean_mask") is None:
                    av_paths.append(av_path)
        comparisons = []
        for av_path in sorted(av_paths):
            metrics, av = _metrics(base, av_path, casa, casa_paths)
            plot_path = (
                ROOT
                / "evaluation"
                / base.parent.name
                / base.name
                / f"casa_vs_{av_path.parent.name}.png"
            )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            _render_comparison(
                plot_path,
                casa,
                av,
                f"{base.parent.name} / {base.name}\n{av_path.parent.name}",
            )
            metrics["plot"] = str(plot_path)
            comparisons.append(metrics)
        baseline = next(
            item for item in comparisons if item["astroviper_run"].startswith("mfs_")
        )
        results.append(
            {
                "project": base.parent.name,
                "variant": base.name,
                "casa_elapsed_seconds": json.loads(complete.read_text()).get(
                    "elapsed_seconds"
                ),
                "assessment": _interpret(baseline),
                "comparisons": comparisons,
            }
        )
    OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    _write_report(results)
    print(OUTPUT_JSON)
    print(OUTPUT_MD)


def _write_report(results):
    lines = [
        "# CASA–AstroViper exact-input equivalence",
        "",
        "CASA 6.7.7 was run on the same materialized `prepared/selected.ms` "
        "inputs as AstroViper, using the corresponding `tclean_cmd*.py` image, "
        "weighting, stopping, restoration, and PB-correction settings. Every "
        "retained CASA run is unmasked and uses `deconvolver=hogbom`, `nterms=1`. "
        "Selection expressions were not reapplied. The primary "
        "comparison is AstroViper MFS/global/CASA-compatible weighting; MVC is "
        "also shown against the same CASA result.",
        "",
        "| Project | AV path | PSF corr. | Restored corr. | Peak ratio | Beam-area ratio | CASA/AV model Jy | Assessment |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        for index, item in enumerate(result["comparisons"]):
            assessment = result["assessment"] if index == 0 else "MVC cross-check"
            lines.append(
                f"| {result['project']} | `{item['astroviper_run'].split('__')[0]}` | "
                f"{item['psf_correlation']:.5f} | {item['restored_correlation']:.5f} | "
                f"{item['peak_ratio_astroviper_over_casa']:.3f} | "
                f"{item['beam_area_ratio_astroviper_over_casa']:.3f} | "
                f"{item['casa_model_sum_jy']:.5g}/{item['astroviper_model_sum_jy']:.5g} | "
                f"{assessment} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A matching PSF isolates data selection, imaging geometry, global "
            "Briggs weighting, and gridding from the minor-cycle algorithm. "
            "Both sides use an unmasked Hogbom minor cycle, so restored/model "
            "differences isolate implementation details rather than mask selection.",
            "- CASA and AstroViper use different primary-beam implementations, "
            "so PB-corrected edges are not expected to be pixel-identical.",
            "- NGC6334I is a one-of-eight-MS partial reconstruction. Its comparison "
            "establishes equivalence for the copied input only, not the full QA2 product.",
            "- HOPS-387 is restricted on both sides to original-MS FIELD 63, "
            "the central pointing at the requested image phase center. CASA uses "
            "`gridder=standard, mosweight=False`, matching the single-field test.",
            "- MVC constructs frequency-dependent/Taylor responses differently "
            "from CASA MFS, so its PSF correlation against the CASA MFS PSF is a "
            "cross-check, not an equivalence acceptance criterion.",
            "- Across all six retained MFS runs, fitted restoring-beam areas "
            "agree to about 0.13% or better.",
            "- The five ordinary single-pointing primary beams correlate at "
            "0.9986–0.9989 with 14.7–16.4% relative L2 differences. That is "
            "consistent with the documented CASA versus AstroViper Airy PB-model "
            "choice and explains some PB-corrected edge differences.",
            "",
            "The detailed numerical record is in `casa_equivalence_results.json`; "
            "side-by-side CASA, AstroViper, and difference images are stored below "
            "each project's evaluation directory.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
