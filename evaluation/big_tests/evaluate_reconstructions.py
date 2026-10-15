#!/usr/bin/env python3
"""Compare CASA FITS products with AstroViper continuum Zarr products."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "evaluation" / "big_tests"
os.environ.setdefault(
    "MPLCONFIGDIR", str(ROOT / "results" / "big_tests" / ".matplotlib")
)

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from astropy.io import fits  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "big_tests"))

from continuum_common import (  # noqa: E402
    BIG_VALIDATION_PROJECTS,
    PROJECTS,
    RESULTS_ROOT,
    astroviper_variants,
    write_manifest,
)

PRODUCTS = {
    "restored": ("image", "SKY_RESTORED"),
    "restored_pbcor": ("image.pbcor", "SKY_RESTORED_PBCOR"),
    "residual": ("residual", "SKY_RESIDUAL"),
    "psf": ("psf", "POINT_SPREAD_FUNCTION"),
    "primary_beam": ("pb", "PRIMARY_BEAM"),
    "model": ("model", "SKY_MODEL"),
}


def _plane(values: Any) -> np.ndarray:
    """Reduce singleton and Taylor-like dimensions to one finite 2-D plane."""
    array = np.asarray(values)
    array = np.squeeze(array)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D image plane, got shape {array.shape}")
    return np.asarray(array, dtype=np.float64)


def _candidate_orientations(array: np.ndarray):
    """Yield the transpose/flip conventions encountered in FITS and Xarray images."""
    seen = set()
    for transpose in (False, True):
        base = array.T if transpose else array
        for flip0 in (False, True):
            for flip1 in (False, True):
                candidate = base
                if flip0:
                    candidate = np.flip(candidate, axis=0)
                if flip1:
                    candidate = np.flip(candidate, axis=1)
                key = (candidate.shape, candidate.strides)
                if key not in seen:
                    seen.add(key)
                    yield (
                        candidate,
                        {
                            "transpose": transpose,
                            "flip_axis_0": flip0,
                            "flip_axis_1": flip1,
                        },
                    )


def _align(reference: np.ndarray, candidate: np.ndarray):
    """Choose the array orientation with the strongest finite-pixel correlation."""
    best = None
    for oriented, transform in _candidate_orientations(candidate):
        if oriented.shape != reference.shape:
            continue
        valid = np.isfinite(reference) & np.isfinite(oriented)
        if np.count_nonzero(valid) < 2:
            continue
        correlation = np.corrcoef(reference[valid], oriented[valid])[0, 1]
        score = -np.inf if not np.isfinite(correlation) else abs(correlation)
        if best is None or score > best[0]:
            best = (score, oriented, transform)
    if best is None:
        raise ValueError(
            f"Cannot align CASA shape {reference.shape} with AstroViper shape "
            f"{candidate.shape}"
        )
    return best[1], best[2]


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Calculate scale-aware reconstruction agreement metrics."""
    valid = np.isfinite(reference) & np.isfinite(candidate)
    casa = reference[valid]
    viper = candidate[valid]
    difference = viper - casa
    reference_rms = float(np.sqrt(np.mean(casa**2)))
    return {
        "n_pixels": int(casa.size),
        "correlation": float(np.corrcoef(casa, viper)[0, 1]),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "normalized_rmse": float(
            np.sqrt(np.mean(difference**2)) / reference_rms
            if reference_rms != 0
            else np.nan
        ),
        "mean_difference": float(np.mean(difference)),
        "casa_rms": reference_rms,
        "astroviper_rms": float(np.sqrt(np.mean(viper**2))),
        "casa_peak": float(np.max(np.abs(casa))),
        "astroviper_peak": float(np.max(np.abs(viper))),
        "flux_ratio": (
            float(np.sum(viper) / np.sum(casa)) if np.sum(casa) != 0 else np.nan
        ),
    }


def _plot(
    reference: np.ndarray,
    candidate: np.ndarray,
    title: str,
    output: Path,
) -> None:
    """Save CASA, AstroViper, and difference planes with robust color limits."""
    valid_values = np.concatenate(
        [reference[np.isfinite(reference)], candidate[np.isfinite(candidate)]]
    )
    lower, upper = np.nanpercentile(valid_values, [1, 99])
    difference = candidate - reference
    difference_limit = np.nanpercentile(np.abs(difference), 99)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, values, label in zip(
        axes[:2], (reference, candidate), ("CASA", "AstroViper"), strict=True
    ):
        image = axis.imshow(values, origin="lower", vmin=lower, vmax=upper)
        axis.set_title(label)
        figure.colorbar(image, ax=axis, shrink=0.8)
    image = axes[2].imshow(
        difference,
        origin="lower",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
    )
    axes[2].set_title("AstroViper - CASA")
    figure.colorbar(image, ax=axes[2], shrink=0.8)
    figure.suptitle(title)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _write_finding(
    output_dir: Path,
    project: str,
    variant: str,
    rows: list[dict[str, Any]],
    manifest_path: Path,
) -> None:
    """Write a per-run finding derived only from current metrics and metadata."""
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    compared = [row for row in rows if row["validation_status"] == "compared"]
    finite_counts = manifest.get("in_memory_finite_counts", {})
    elapsed = manifest.get("elapsed_seconds")
    iteration = manifest.get("iteration_control_params", {})
    deconvolver = manifest.get(
        "deconvolver",
        manifest.get("source_tclean_parameters", {}).get("deconvolver", "unknown"),
    )
    nterms = manifest.get(
        "nterms",
        manifest.get("image_params", {}).get("nterms", "unknown"),
    )

    lines = [
        f"# {project} {variant} continuum validation",
        "",
        "> Generated by `evaluate_reconstructions.py`; rerunning the evaluator "
        "replaces this file.",
        "",
        f"Compared {len(compared)} of {len(PRODUCTS)} supported products with "
        "the available CASA reference.",
    ]
    if finite_counts:
        all_finite = all(int(count) > 0 for count in finite_counts.values())
        lines.append(
            "All recorded AstroViper products contain finite pixels."
            if all_finite
            else "At least one recorded AstroViper product has no finite pixels."
        )
    if elapsed is not None:
        lines.extend(
            [
                "",
                "## Run configuration",
                "",
                f"- AstroViper elapsed time: `{float(elapsed):.3f} s`",
                f"- Deconvolver: `{deconvolver}`",
                f"- Taylor terms: `{nterms}`",
                "- Requested iterations: "
                f"`{iteration.get('max_iter', iteration.get('niter', 'unknown'))}`",
                "- Major-cycle limit: "
                f"`{iteration.get('max_cycles', iteration.get('nmajor', 'unknown'))}`",
                "- Minor-cycle limit: "
                f"`{iteration.get('max_iter_per_cycle', iteration.get('cycleniter', 'unknown'))}`",
                f"- Stopping threshold: `{iteration.get('threshold', 'unknown')} Jy`",
            ]
        )

    lines.extend(
        [
            "",
            "## CASA comparison",
            "",
            "| Product | Status | Correlation | Normalized RMSE | Flux ratio |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['product']} | {row['validation_status']} | "
            f"{row['correlation']:.6g} | {row['normalized_rmse']:.6g} | "
            f"{row['flux_ratio']:.6g} |"
        )
    lines.extend(
        [
            "",
            "These are comparison measurements, not universal pass/fail thresholds. "
            f"AstroViper used the configured `{deconvolver}` deconvolver without "
            "CASA auto-multithresh.",
        ]
    )
    (output_dir / "finding.md").write_text("\n".join(lines) + "\n")


def evaluate_variant(
    project: str, variant: str, deconvolver: str
) -> list[dict[str, Any]]:
    """Evaluate every corresponding product available for one reconstruction."""
    run_dir = RESULTS_ROOT / project / variant
    casa_dir = run_dir / "casa"
    viper_dir = run_dir / "astroviper" / deconvolver
    viper_store = viper_dir / "continuum.img.zarr"
    if not viper_store.exists():
        raise FileNotFoundError(viper_store)
    image = xr.open_zarr(viper_store)
    output_variant = f"{variant}_{deconvolver}"
    output_dir = EVALUATION_ROOT / project / output_variant
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for product, (fits_stem, variable) in PRODUCTS.items():
        fits_path = casa_dir / f"{fits_stem}.fits"
        if not fits_path.exists() or variable not in image:
            continue
        reference = _plane(fits.getdata(fits_path))
        candidate = _plane(image[variable].values)
        if not np.any(np.isfinite(candidate)):
            rows.append(
                {
                    "project": project,
                    "variant": output_variant,
                    "product": product,
                    "validation_status": "no_finite_astroviper_pixels",
                    "casa_fits": str(fits_path),
                    "astroviper_store": str(viper_store),
                    "astroviper_variable": variable,
                    "orientation": "",
                    **{
                        key: np.nan
                        for key in (
                            "n_pixels",
                            "correlation",
                            "rmse",
                            "normalized_rmse",
                            "mean_difference",
                            "casa_rms",
                            "astroviper_rms",
                            "casa_peak",
                            "astroviper_peak",
                            "flux_ratio",
                        )
                    },
                }
            )
            continue
        candidate, transform = _align(reference, candidate)
        metrics = _metrics(reference, candidate)
        row = {
            "project": project,
            "variant": output_variant,
            "product": product,
            "casa_fits": str(fits_path),
            "astroviper_store": str(viper_store),
            "astroviper_variable": variable,
            "validation_status": "compared",
            "orientation": json.dumps(transform, sort_keys=True),
            **metrics,
        }
        rows.append(row)
        _plot(
            reference,
            candidate,
            f"{project} {output_variant}: {product}",
            output_dir / f"{product}.png",
        )

    write_manifest(output_dir / "metrics.json", {"comparisons": rows})
    _write_finding(
        output_dir,
        project,
        output_variant,
        rows,
        viper_dir / "complete.json",
    )
    return rows


def _load_all_metrics() -> list[dict[str, Any]]:
    """Load every per-variant report so a selected run keeps the global summary."""
    rows = []
    for metrics_path in sorted(EVALUATION_ROOT.glob("*/*_hogbom/metrics.json")):
        rows.extend(json.loads(metrics_path.read_text()).get("comparisons", []))
    for metrics_path in sorted(EVALUATION_ROOT.glob("*/*_mtmfs/metrics.json")):
        rows.extend(json.loads(metrics_path.read_text()).get("comparisons", []))
    return rows


def write_reports(rows: list[dict[str, Any]]) -> None:
    """Write machine-readable CSV and a compact Markdown summary."""
    output_csv = EVALUATION_ROOT / "summary.csv"
    output_md = EVALUATION_ROOT / "summary.md"
    if rows:
        with output_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Continuum reconstruction comparison",
        "",
        "| Project | Variant | Product | Status | Correlation | Normalized RMSE | Flux ratio |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['project']} | {row['variant']} | {row['product']} | "
            f"{row['validation_status']} | "
            f"{row['correlation']:.6g} | {row['normalized_rmse']:.6g} | "
            f"{row['flux_ratio']:.6g} |"
        )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Evaluate one project or all completed continuum runs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=(*PROJECTS, "all"), default="all", nargs="?")
    args = parser.parse_args()
    projects = BIG_VALIDATION_PROJECTS if args.project == "all" else (args.project,)
    rows = []
    for project in projects:
        for variant, deconvolver, _ in astroviper_variants(project):
            try:
                rows.extend(evaluate_variant(project, variant, deconvolver))
            except FileNotFoundError as error:
                print(
                    f"Skipping incomplete {project}/{variant}/{deconvolver}: {error}"
                )
    write_reports(_load_all_metrics())


if __name__ == "__main__":
    main()
