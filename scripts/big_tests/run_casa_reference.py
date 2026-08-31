#!/usr/bin/env python3
"""Run the original continuum tclean configurations and export comparison FITS."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from continuum_common import (
    DATA_ROOT,
    PROJECTS,
    RESULTS_ROOT,
    project_variants,
    resolve_available_config,
    write_manifest,
)


def _export_products(prefix: Path, result_dir: Path, overwrite: bool) -> dict[str, str]:
    """Export ordinary or Taylor-term-zero CASA products to stable FITS names."""
    from casatasks import exportfits

    candidates = {
        "image": ("image", "image.tt0"),
        "image.pbcor": ("image.pbcor", "image.tt0.pbcor"),
        "residual": ("residual", "residual.tt0"),
        "psf": ("psf", "psf.tt0"),
        "pb": ("pb", "pb.tt0"),
        "model": ("model", "model.tt0"),
    }
    products = {}
    for output_stem, suffixes in candidates.items():
        image = next(
            (
                Path(f"{prefix}.{suffix}")
                for suffix in suffixes
                if Path(f"{prefix}.{suffix}").exists()
            ),
            None,
        )
        if image is None:
            continue
        fits_path = result_dir / f"{output_stem}.fits"
        if fits_path.exists() and not overwrite:
            products[output_stem] = str(fits_path)
            continue
        exportfits(
            imagename=str(image),
            fitsimage=str(fits_path),
            overwrite=overwrite,
            dropdeg=True,
        )
        products[output_stem] = str(fits_path)
    return products


def _remove_previous_products(prefix: Path, result_dir: Path) -> None:
    """Remove products owned by one CASA configuration before an overwrite run."""
    for path in prefix.parent.glob(prefix.name + "*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in result_dir.glob("*.fits"):
        path.unlink()


def run_project(project: str, overwrite: bool) -> None:
    """Run every MFS reference configuration belonging to one project."""
    from casatasks import tclean

    data_dir = DATA_ROOT / project
    for variant, original in project_variants(project):
        result_dir = RESULTS_ROOT / project / variant / "casa"
        result_dir.mkdir(parents=True, exist_ok=True)
        prefix = result_dir / Path(original["imagename"]).name
        marker = result_dir / "complete.json"
        if marker.exists() and not overwrite:
            manifest = json.loads(marker.read_text())
            manifest["fits_products"] = _export_products(prefix, result_dir, False)
            write_manifest(marker, manifest)
            print(f"Reused completed CASA result {project}/{variant}")
            continue

        if overwrite:
            _remove_previous_products(prefix, result_dir)

        params, omitted_vis = resolve_available_config(original, data_dir)
        params["imagename"] = str(prefix)
        params["parallel"] = False
        # The archived commands are restart-stage commands.  A clean validation
        # directory needs the initial residual and PSF to be calculated.
        params["restart"] = False
        params["calcres"] = True
        params["calcpsf"] = True

        print(f"Running CASA tclean for {project}/{variant}")
        tclean(**params)

        fits_products = _export_products(prefix, result_dir, overwrite)

        write_manifest(
            marker,
            {
                "project": project,
                "variant": variant,
                "original_parameters": original,
                "effective_parameters": params,
                "image_prefix": str(prefix),
                "fits_products": fits_products,
                "omitted_unavailable_measurement_sets": omitted_vis,
                "casa_log": os.environ.get("CASALOGFILE"),
            },
        )


def main() -> None:
    """Parse command-line arguments and run selected CASA references."""
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=(*PROJECTS, "all"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    projects = PROJECTS if args.project == "all" else (args.project,)
    for project in projects:
        run_project(project, args.overwrite)


if __name__ == "__main__":
    main()
