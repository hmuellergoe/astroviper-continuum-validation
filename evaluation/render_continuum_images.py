#!/usr/bin/env python3
"""Render and numerically inspect completed continuum demonstrator images."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/astroviper-continuum-render-matplotlib")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi


def _plane(dataset, variable, order=None):
    """Load one Stokes-I image plane as a two-dimensional NumPy array."""
    array = dataset[variable]
    if "polarization" in array.dims:
        array = array.sel(polarization="I")
    indexers = {dimension: 0 for dimension in array.dims if dimension not in ("l", "m")}
    if order is not None:
        for dimension in ("taylor_term", "psf_taylor_order", "frequency"):
            if dimension in array.dims:
                indexers[dimension] = order
                break
    return np.asarray(array.isel(indexers).values, dtype=np.float64).squeeze()


def _finite_peak(array):
    """Return the maximum finite absolute pixel value."""
    values = np.asarray(array)
    values = values[np.isfinite(values)]
    return float(np.max(np.abs(values))) if values.size else np.nan


def _relative_rms(left, right):
    """Return RMS(left-right) divided by RMS(left) on common finite pixels."""
    mask = np.isfinite(left) & np.isfinite(right)
    if not np.any(mask):
        return np.nan
    denominator = np.sqrt(np.mean(np.square(left[mask])))
    return float(np.sqrt(np.mean(np.square(left[mask] - right[mask]))) / denominator)


def _correlation(left, right):
    """Return Pearson correlation on common finite pixels."""
    mask = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 2:
        return np.nan
    left_values = left[mask]
    right_values = right[mask]
    if np.std(left_values) == 0.0 or np.std(right_values) == 0.0:
        return np.nan
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _display_norm(arrays):
    """Build one robust symmetric logarithmic normalization for several planes."""
    finite = np.concatenate(
        [np.abs(array[np.isfinite(array)]).ravel() for array in arrays]
    )
    nonzero = finite[finite > 0.0]
    if nonzero.size == 0:
        return SymLogNorm(linthresh=1.0, vmin=-1.0, vmax=1.0)
    vmax = float(np.percentile(nonzero, 99.95))
    vmax = max(vmax, float(np.max(nonzero)) * 0.05)
    linthresh = max(float(np.percentile(nonzero, 50.0)), vmax * 1.0e-4)
    return SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)


def _extent(dataset):
    """Return image bounds in arcseconds relative to the image center."""
    longitude = np.asarray(dataset.l.values, dtype=np.float64) * RAD_TO_ARCSEC
    latitude = np.asarray(dataset.m.values, dtype=np.float64) * RAD_TO_ARCSEC
    longitude -= 0.5 * (longitude[0] + longitude[-1])
    latitude -= 0.5 * (latitude[0] + latitude[-1])
    return [
        float(longitude[0]),
        float(longitude[-1]),
        float(latitude[0]),
        float(latitude[-1]),
    ]


def _sort_key(record):
    """Order configurations by MFS/MVC, global/local, and CASA/proper weights."""
    experiment = record["manifest"]["experiment"]
    return (
        {"mfs": 0, "mvc": 1}.get(experiment.get("specmode"), 9),
        {"global": 0, "local": 1}.get(experiment.get("weighting_scope"), 9),
        {"casa": 0, "proper": 1}.get(experiment.get("weight_implementation"), 9),
        record["path"].name,
    )


def _label(record):
    """Return a compact semantic plot label."""
    experiment = record["manifest"]["experiment"]
    return " / ".join(
        (
            str(experiment.get("specmode", "?")).upper(),
            str(experiment.get("weighting_scope", "?")),
            str(experiment.get("weight_implementation", "?")),
        )
    )


def _render_restored(records, output_path, project, variant):
    """Render all restored Taylor-zero planes with one common intensity scale."""
    arrays = [record["restored"] for record in records]
    norm = _display_norm(arrays)
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    image = None
    for axis, record in zip(axes.flat, records, strict=False):
        image = axis.imshow(
            record["restored"].T,
            origin="lower",
            extent=record["extent"],
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        axis.set_title(_label(record), fontsize=10)
        axis.set_xlabel("Δl (arcsec)")
        axis.set_ylabel("Δm (arcsec)")
    for axis in axes.flat[len(records) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="Restored Taylor-0 value")
    figure.suptitle(
        f"{project} / {variant}: AstroVIPER restored continuum", fontsize=14
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_reference_products(record, output_path, project, variant):
    """Render model, residual, restored, and PB-corrected reference products."""
    products = (
        ("Model Taylor 0", record["model"]),
        ("Residual Taylor 0", record["residual"]),
        ("Restored Taylor 0", record["restored"]),
        ("Restored PB-corrected", record["pbcor"]),
    )
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    for axis, (title, array) in zip(axes, products, strict=True):
        image = axis.imshow(
            array.T,
            origin="lower",
            extent=record["extent"],
            cmap="RdBu_r",
            norm=_display_norm([array]),
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("Δl (arcsec)")
        axis.set_ylabel("Δm (arcsec)")
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(
        f"{project} / {variant}: {_label(record)} reference products", fontsize=14
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _render_center_zoom(records, output_path, project, variant):
    """Render a beam-scaled center crop to expose compact target structure."""
    radius = min(
        min(records[0]["restored"].shape) // 4,
        max(16, int(np.ceil(12.0 * records[0]["beam_fwhm_pixels"]))),
    )
    center = np.asarray(records[0]["restored"].shape) // 2
    slices = (
        slice(
            max(0, center[0] - radius),
            min(records[0]["restored"].shape[0], center[0] + radius),
        ),
        slice(
            max(0, center[1] - radius),
            min(records[0]["restored"].shape[1], center[1] + radius),
        ),
    )
    arrays = [record["restored"][slices] for record in records]
    norm = _display_norm(arrays)
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    image = None
    for axis, record, array in zip(axes.flat, records, arrays, strict=False):
        full_shape = record["restored"].shape
        longitude = np.linspace(record["extent"][0], record["extent"][1], full_shape[0])
        latitude = np.linspace(record["extent"][2], record["extent"][3], full_shape[1])
        crop_extent = [
            longitude[slices[0]][0],
            longitude[slices[0]][-1],
            latitude[slices[1]][0],
            latitude[slices[1]][-1],
        ]
        image = axis.imshow(
            array.T,
            origin="lower",
            extent=crop_extent,
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        axis.set_title(_label(record), fontsize=10)
        axis.set_xlabel("Δl (arcsec)")
        axis.set_ylabel("Δm (arcsec)")
    for axis in axes.flat[len(records) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.82, label="Restored Taylor-0 value")
    figure.suptitle(
        f"{project} / {variant}: central continuum ({radius:.0f}-pixel radius)",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _pairwise(records):
    """Measure available CASA/proper, local/global, and MFS/MVC differences."""
    by_key = {}
    for record in records:
        experiment = record["manifest"]["experiment"]
        key = (
            experiment.get("specmode"),
            experiment.get("weighting_scope"),
            experiment.get("weight_implementation"),
        )
        by_key[key] = record

    comparisons = []
    requested_pairs = []
    for specmode in ("mfs", "mvc"):
        for scope in ("global", "local"):
            requested_pairs.append(
                (
                    "CASA versus proper weights",
                    (specmode, scope, "casa"),
                    (specmode, scope, "proper"),
                )
            )
    for specmode in ("mfs", "mvc"):
        for implementation in ("casa", "proper"):
            requested_pairs.append(
                (
                    "global versus local weighting",
                    (specmode, "global", implementation),
                    (specmode, "local", implementation),
                )
            )
    for scope in ("global", "local"):
        for implementation in ("casa", "proper"):
            requested_pairs.append(
                (
                    "MFS versus MVC",
                    ("mfs", scope, implementation),
                    ("mvc", scope, implementation),
                )
            )

    for category, left_key, right_key in requested_pairs:
        if left_key not in by_key or right_key not in by_key:
            continue
        left = by_key[left_key]
        right = by_key[right_key]
        pb_mask = (left["primary_beam"] >= 0.2) & (right["primary_beam"] >= 0.2)
        left_pb = np.where(pb_mask, left["restored"], np.nan)
        right_pb = np.where(pb_mask, right["restored"], np.nan)
        comparisons.append(
            {
                "category": category,
                "left": "/".join(left_key),
                "right": "/".join(right_key),
                "restored_correlation": _correlation(
                    left["restored"], right["restored"]
                ),
                "restored_relative_rms": _relative_rms(
                    left["restored"], right["restored"]
                ),
                "restored_pb20_correlation": _correlation(left_pb, right_pb),
                "restored_pb20_relative_rms": _relative_rms(left_pb, right_pb),
                "residual_relative_rms": _relative_rms(
                    left["residual"], right["residual"]
                ),
            }
        )
    return comparisons


def _analyze_variant(project_dir, output_root):
    """Load, render, and summarize one project/variant directory."""
    project = project_dir.parent.name
    variant = project_dir.name
    records = []
    for manifest_path in sorted((project_dir / "astroviper").glob("*/complete.json")):
        path = manifest_path.parent
        image_path = path / "continuum.img.zarr"
        if not image_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        dataset = xr.open_zarr(image_path)
        model = _plane(dataset, "SKY_MODEL", 0)
        residual = _plane(dataset, "SKY_RESIDUAL", 0)
        restored = _plane(dataset, "SKY_RESTORED", 0)
        pbcor = _plane(dataset, "SKY_RESTORED_PBCOR")
        primary_beam = _plane(dataset, "PRIMARY_BEAM", 0)
        psf = _plane(dataset, "POINT_SPREAD_FUNCTION", 0)
        beam = np.asarray(
            dataset.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(polarization="I")
            .isel(time=0)
            .values,
            dtype=np.float64,
        )
        pixel_scale = np.sqrt(
            abs(float(np.diff(dataset.l.values[:2])[0]))
            * abs(float(np.diff(dataset.m.values[:2])[0]))
        )
        threshold = float(
            manifest.get("iteration_control_params", {}).get("threshold", np.nan)
        )
        residual_peak = _finite_peak(residual)
        records.append(
            {
                "path": path,
                "manifest": manifest,
                "extent": _extent(dataset),
                "model": model,
                "residual": residual,
                "restored": restored,
                "pbcor": pbcor,
                "primary_beam": primary_beam,
                "psf": psf,
                "beam_fwhm_pixels": float(beam[0] / pixel_scale),
                "metrics": {
                    "configuration": path.name,
                    "model_nonzero_pixels": int(np.count_nonzero(model)),
                    "model_peak": _finite_peak(model),
                    "model_absolute_sum": float(np.nansum(np.abs(model))),
                    "residual_peak": residual_peak,
                    "clean_threshold": threshold,
                    "residual_peak_over_threshold": (
                        float(residual_peak / threshold) if threshold > 0.0 else np.nan
                    ),
                    "restored_peak": _finite_peak(restored),
                    "restored_minus_residual_relative_rms": _relative_rms(
                        restored, residual
                    ),
                    "finite_restored_fraction": float(np.mean(np.isfinite(restored))),
                    "finite_pbcor_fraction": float(np.mean(np.isfinite(pbcor))),
                    "primary_beam_min": float(np.nanmin(primary_beam)),
                    "primary_beam_max": float(np.nanmax(primary_beam)),
                    "psf_peak": float(np.nanmax(psf)),
                },
            }
        )
        dataset.close()

    records.sort(key=_sort_key)
    if not records:
        raise FileNotFoundError(f"No completed image stores found below {project_dir}")

    output_dir = output_root / project / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    _render_restored(
        records,
        output_dir / "continuum_restored.png",
        project,
        variant,
    )
    reference = next(
        (
            record
            for record in records
            if record["manifest"]["experiment"].get("specmode") == "mfs"
            and record["manifest"]["experiment"].get("weighting_scope") == "global"
            and record["manifest"]["experiment"].get("weight_implementation") == "casa"
        ),
        records[0],
    )
    _render_reference_products(
        reference,
        output_dir / "continuum_reference_products.png",
        project,
        variant,
    )
    _render_center_zoom(
        records,
        output_dir / "continuum_center_zoom.png",
        project,
        variant,
    )
    diagnostics = {
        "project": project,
        "variant": variant,
        "reference_configuration": reference["path"].name,
        "configurations": [record["metrics"] for record in records],
        "pairwise_comparisons": _pairwise(records),
    }
    (output_dir / "continuum_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=True) + "\n"
    )
    print(f"Rendered {project}/{variant} to {output_dir}")
    del records
    gc.collect()


def main():
    """Render all completed variants below a demonstrator results directory."""
    parser = argparse.ArgumentParser()
    demonstrator_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--results-root", type=Path, default=demonstrator_root / "results"
    )
    parser.add_argument(
        "--output-root", type=Path, default=demonstrator_root / "evaluation"
    )
    args = parser.parse_args()
    variants = sorted(
        path.parents[2]
        for path in args.results_root.glob("*/*/astroviper/*/complete.json")
    )
    for project_dir in sorted(set(variants)):
        _analyze_variant(project_dir, args.output_root)


if __name__ == "__main__":
    main()
