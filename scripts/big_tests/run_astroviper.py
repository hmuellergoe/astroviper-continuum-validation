#!/usr/bin/env python3
"""Convert selected MSv2 inputs and run AstroViper continuum imaging."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from continuum_common import (
    PROJECTS,
    RESULTS_ROOT,
    parse_cell_radians,
    parse_jy,
    project_variants,
    write_manifest,
)


def _convert_ms(selected_ms: Path, ps_store: Path, overwrite: bool) -> None:
    """Convert one prepared MSv2 to an XRADIO Processing Set when needed."""
    from xradio.measurement_set import (
        convert_msv2_to_processing_set,
        open_processing_set,
    )

    if ps_store.exists() and not overwrite:
        try:
            if len(open_processing_set(str(ps_store))) > 0:
                return
        except Exception:
            if not overwrite:
                raise RuntimeError(
                    f"Incomplete Processing Set {ps_store}; rerun with --overwrite"
                ) from None

    convert_msv2_to_processing_set(
        in_file=str(selected_ms),
        out_file=str(ps_store),
        partition_scheme=[],
        main_chunksize=0.25,
        with_pointing=False,
        use_table_iter=False,
        parallel_mode="none",
        persistence_mode="w" if ps_store.exists() else "w-",
    )


def run_project(
    project: str,
    overwrite: bool,
    niter_override: int | None,
    nmajor: int,
    n_chunks: int | None,
    cores: int,
    memory_limit: str,
) -> None:
    """Run all continuum configurations for one project."""
    from astroviper.distributed_applications.imaging import image_continuum_single_field
    from toolviper.dask.client import local_client
    from xradio.measurement_set import open_processing_set

    for variant, config in project_variants(project):
        result_dir = RESULTS_ROOT / project / variant / "astroviper"
        result_dir.mkdir(parents=True, exist_ok=True)
        selected_ms = RESULTS_ROOT / project / variant / "prepared" / "selected.ms"
        if not selected_ms.exists():
            raise FileNotFoundError(
                f"{selected_ms} is missing; run prepare_selected_ms.py first"
            )
        ps_store = result_dir / "selected.ps.zarr"
        image_store = result_dir / "continuum.img.zarr"
        marker = result_dir / "complete.json"
        if marker.exists() and image_store.exists() and not overwrite:
            print(f"Skipping completed AstroViper result {project}/{variant}")
            continue
        if overwrite and marker.exists():
            marker.unlink()

        _convert_ms(selected_ms, ps_store, overwrite)
        ps_xdt = open_processing_set(str(ps_store))
        combined_field = ps_xdt.xr_ps.get_combined_field_and_source_xds()
        center_field = combined_field.attrs["center_field_name"]
        phase_direction = combined_field.FIELD_PHASE_CENTER_DIRECTION.sel(
            field_name=center_field
        ).values
        frequency_coords = ps_xdt.xr_ps.get_freq_axis().values
        cell = parse_cell_radians(config["cell"])

        image_params = {
            "image_size": list(config["imsize"]),
            "cell_size": np.array([-cell, cell]),
            "phase_direction": phase_direction,
            "frequency_coords": frequency_coords,
            "polarization_coords": ["I"],
            "time_coords": [0],
            "fft_padding": 1.2,
            "cpp_gridder": True,
            "nterms": int(config.get("nterms", 2)),
            "reference_frequency_hz": float(np.mean(frequency_coords)),
        }
        imaging_weights_params = {
            "weighting": config.get("weighting", "natural"),
            "robust": float(config.get("robust", 0.5)),
            "casa_weighting_implementation": True,
        }
        effective_niter = (
            int(config.get("niter", 0)) if niter_override is None else niter_override
        )
        iteration_control_params = {
            "niter": effective_niter,
            "nmajor": nmajor,
            "threshold": parse_jy(config.get("threshold", 0.0)),
            "gain": float(config.get("gain", 0.1)),
            "cyclefactor": float(config.get("cyclefactor", 1.0)),
            "cycleniter": int(config.get("cycleniter", 50)),
            "minpsffraction": float(config.get("minpsffraction", 0.05)),
            "maxpsffraction": float(config.get("maxpsffraction", 0.8)),
        }

        print(f"Running AstroViper continuum imaging for {project}/{variant}")
        started = time.time()
        client = local_client(cores=cores, memory_limit=memory_limit)
        in_memory_store = result_dir / "continuum.in_memory.img.zarr"
        in_memory_finite_counts = {}
        try:
            result = image_continuum_single_field(
                ps_store=str(ps_store),
                image_store=str(image_store),
                image_params=image_params,
                imaging_weights_params=imaging_weights_params,
                iteration_control_params=iteration_control_params,
                gridder="prolate_spheroidal",
                deconvolver="hogbom",
                pbcor=bool(config.get("pbcor", False)),
                pblimit=float(config.get("pblimit", 0.2)),
                specmode="mfs",
                scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
                field_name=config.get("field"),
                image_data_variables_keep=[
                    "sky_model",
                    "sky_residual",
                    "sky_restored",
                    "point_spread_function",
                    "primary_beam",
                    "beam_fit_params_point_spread_function",
                ],
                processing_set_data_group_name="base",
                processing_function_threads=1,
                n_chunks=n_chunks,
                overwrite=overwrite,
                memory_mode="in_memory",
                fft_backend="scipy",
                restore=True,
                compute_backend="dask",
                reduce_mode="tree",
                reduce_n_batch=2,
            )
            in_memory_image = result["image"]
            diagnostic_variables = [
                name
                for name in (
                    "SKY_MODEL",
                    "SKY_RESIDUAL",
                    "SKY_RESTORED",
                    "SKY_RESTORED_PBCOR",
                    "POINT_SPREAD_FUNCTION",
                    "PRIMARY_BEAM",
                )
                if name in in_memory_image
            ]
            diagnostic_image = in_memory_image[diagnostic_variables]
            in_memory_finite_counts = {
                name: int(np.count_nonzero(np.isfinite(diagnostic_image[name].values)))
                for name in diagnostic_variables
            }
            diagnostic_image.to_zarr(in_memory_store, mode="w")
        finally:
            client.close()

        write_manifest(
            marker,
            {
                "project": project,
                "variant": variant,
                "source_tclean_parameters": config,
                "processing_set": str(ps_store),
                "image_store": str(image_store),
                "in_memory_diagnostic_store": str(in_memory_store),
                "in_memory_finite_counts": in_memory_finite_counts,
                "image_params": image_params,
                "imaging_weights_params": imaging_weights_params,
                "iteration_control_params": iteration_control_params,
                "n_chunks": n_chunks,
                "elapsed_seconds": time.time() - started,
                "return_keys": sorted(result),
                "limitations": [
                    "AstroViper currently uses Hogbom without CASA auto-multithresh.",
                    "The selected MS is materialized before XRADIO conversion.",
                ],
            },
        )


def main(default_project: str | None = None) -> None:
    """Parse run controls and execute one or all project configurations."""
    parser = argparse.ArgumentParser()
    if default_project is None:
        parser.add_argument("project", choices=(*PROJECTS, "all"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--niter", type=int, default=None)
    parser.add_argument("--nmajor", type=int, default=20)
    parser.add_argument("--n-chunks", type=int, default=None)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()
    selected = default_project or args.project
    projects = PROJECTS if selected == "all" else (selected,)
    for project in projects:
        run_project(
            project,
            args.overwrite,
            args.niter,
            args.nmajor,
            args.n_chunks,
            args.cores,
            args.memory_limit,
        )


if __name__ == "__main__":
    main()
