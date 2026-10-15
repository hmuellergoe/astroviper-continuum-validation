"""Generate RADPS/AstroVIPER continuum major-cycle products.

This script converts each copied MSv2 to an xradio Processing Set, runs one
zero-iteration continuum cycle in MFS and/or MVC, and exports Taylor residual,
PSF, and PB arrays to a neutral NPZ format.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(HARNESS_ROOT / "scripts"))
from cases import (  # noqa: E402
    CASES,
    CLEANING_CASE,
    WEIGHTING_CASES,
    cleaning_suffix,
    selected_cases,
    selected_weightings,
    weighting_suffix,
)
from iteration_limits import casa_to_astroviper_iteration_limit  # noqa: E402


def convert_ms(ms_path: Path, ps_path: Path, overwrite: bool) -> None:
    from xradio.measurement_set import convert_msv2_to_processing_set

    if ps_path.exists() and not overwrite:
        return
    if ps_path.exists():
        shutil.rmtree(ps_path)
    convert_msv2_to_processing_set(
        in_file=str(ms_path),
        out_file=str(ps_path),
        partition_scheme=[],
        main_chunksize=0.01,
        pointing_chunksize=0.00001,
        pointing_interpolate=True,
        ephemeris_interpolate=True,
        use_table_iter=False,
        parallel_mode="none",
        persistence_mode="w-",
    )


def phase_direction_from_ps(ps_path: Path):
    from xradio.measurement_set import open_processing_set

    ps_xdt = open_processing_set(str(ps_path))
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    center_name = combined.attrs["center_field_name"]
    direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(field_name=center_name)
    return np.asarray(direction.values), np.asarray(ps_xdt.xr_ps.get_freq_axis().values)


def require_one_processing_set_field(ps_path: Path) -> str:
    """Return the sole field name, rejecting multi-field converted fixtures."""
    from xradio.measurement_set import open_processing_set

    ps_xdt = open_processing_set(str(ps_path))
    field_names = set()
    for child_name, child_xdt in ps_xdt.items():
        if "field_name" not in child_xdt.coords:
            raise RuntimeError(
                f"Processing Set child {child_name!r} has no field_name coordinate."
            )
        field_names.update(str(value) for value in np.unique(child_xdt.field_name))

    if len(field_names) != 1:
        raise RuntimeError(
            "The off-axis single-field test requires exactly one field after "
            f"conversion; found {sorted(field_names)}."
        )
    return field_names.pop()


def export_products(result: dict, destination: Path) -> None:
    image = result["image"]
    required = ("SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM")
    missing = [name for name in required if name not in image]
    if missing:
        raise RuntimeError(f"RADPS result is missing products: {missing}")

    residual = image["SKY_RESIDUAL"]
    psf = image["POINT_SPREAD_FUNCTION"]
    pb = image["PRIMARY_BEAM"]

    def scalar_i(data_array):
        indexers = {}
        if "time" in data_array.dims:
            indexers["time"] = 0
        if "polarization" in data_array.dims:
            indexers["polarization"] = 0
        return np.asarray(data_array.isel(indexers).values)

    destination.parent.mkdir(parents=True, exist_ok=True)
    products = dict(
        residual_t0=scalar_i(residual.isel(taylor_term=0)),
        residual_t1=scalar_i(residual.isel(taylor_term=1)),
        psf_t0=scalar_i(psf.isel(psf_taylor_order=0)),
        psf_t1=scalar_i(psf.isel(psf_taylor_order=1)),
        psf_t2=scalar_i(psf.isel(psf_taylor_order=2)),
        pb_t0=scalar_i(pb.isel(frequency=0)),
    )
    if "SKY_MODEL" in image:
        model = image["SKY_MODEL"]
        products.update(
            model_t0=scalar_i(model.isel(taylor_term=0)),
            model_t1=scalar_i(model.isel(taylor_term=1)),
        )
    np.savez_compressed(destination, **products)


def validate_persisted_products(returned_image, persisted_image) -> None:
    """Require the public image store to match finite in-memory products."""
    required = ["SKY_RESIDUAL", "POINT_SPREAD_FUNCTION", "PRIMARY_BEAM"]
    if "SKY_MODEL" in returned_image:
        required.append("SKY_MODEL")
    for name in required:
        if name not in persisted_image:
            raise RuntimeError(f"Persisted RADPS image is missing {name}.")
        if not np.isfinite(persisted_image[name].values).any():
            raise RuntimeError(f"Persisted RADPS image {name} has no finite values.")
        if persisted_image[name].dims != returned_image[name].dims:
            raise RuntimeError(
                f"Persisted {name} dimensions {persisted_image[name].dims} differ "
                f"from in-memory dimensions {returned_image[name].dims}."
            )
        np.testing.assert_allclose(
            persisted_image[name].values,
            returned_image[name].values,
            equal_nan=True,
        )


def run_case(
    case_name: str,
    config: dict,
    mode: str,
    weighting_name: str,
    weighting_config: dict,
    args,
):
    from astroviper.distributed_applications.imaging import image_continuum_single_field

    ms_path = args.data_dir / config["ms"]
    ps_path = args.work_dir / f"{case_name}.ps.zarr"
    convert_ms(ms_path, ps_path, args.overwrite_conversion)
    selected_field_name = config["field_name"]
    if config.get("require_single_field", False):
        selected_field_name = require_one_processing_set_field(ps_path)
    phase_direction, frequencies = phase_direction_from_ps(ps_path)
    if config["phasecenter"] is not None:
        raise NotImplementedError(
            "The current harness cases derive their image center from the "
            "Measurement Set; explicit phase-center shifts require a separate test."
        )

    suffix = weighting_suffix(weighting_name)
    clean_suffix = cleaning_suffix(args.clean)
    image_store = (
        args.work_dir / f"radps_{case_name}_{mode}{suffix}{clean_suffix}.img.zarr"
    )
    image_params = {
        "image_size": [config["imsize"], config["imsize"]],
        "cell_size": np.asarray([-1.0, 1.0])
        * config["cell_arcsec"]
        * np.pi
        / (180.0 * 3600.0),
        "phase_direction": phase_direction,
        "frequency_coords": frequencies,
        "polarization_coords": ["I"],
        "time_coords": [0],
        "spectral_reference": "topo",
        "fft_padding": 1.2,
        "cpp_gridder": True,
        "reference_frequency_hz": config["reffreq_hz"],
        "reference_frequency": config["reffreq_hz"],
        "nterms": 2,
    }
    imaging_weights_params = {
        "weighting": weighting_config["weighting"],
        "weighting_scope": weighting_config["weighting_scope"],
        "casa_weighting_implementation": True,
    }
    if weighting_config["robust"] is not None:
        imaging_weights_params["robust"] = weighting_config["robust"]
    iteration_control_params = {
        "max_iter": casa_to_astroviper_iteration_limit(
            CLEANING_CASE["niter"] if args.clean else 0
        ),
        "max_cycles": CLEANING_CASE["nmajor"] if args.clean else 0,
        "threshold": CLEANING_CASE["threshold_jy"] if args.clean else 1.0e30,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.0,
        # CASA leaves cycleniter at its unlimited default in this comparison.
        "max_iter_per_cycle": casa_to_astroviper_iteration_limit(-1),
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "threshold_sigma": 0.0,
    }

    print(f"Running RADPS {case_name}/{mode}/{weighting_name}: {config['description']}")
    result = image_continuum_single_field(
        ps_store=str(ps_path),
        image_store=str(image_store),
        image_params=image_params,
        imaging_weights_params=imaging_weights_params,
        iteration_control_params=iteration_control_params,
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        restore=False,
        pbcor=False,
        pblimit=config["pblimit"],
        specmode=mode,
        field_name=selected_field_name,
        # XRADIO represents the simulator metadata differently in these two
        # regression datasets, so select the intent recorded for this case.
        scan_intents=config["scan_intents"],
        image_data_variables_keep=[
            "sky_model",
            "sky_residual",
            "point_spread_function",
            "primary_beam",
            "beam_fit_params_point_spread_function",
        ],
        processing_set_data_group_name="base",
        single_precision_image=False,
        processing_function_threads=1,
        n_chunks=args.n_chunks,
        overwrite=True,
        memory_mode="in_memory",
        vizualize_graph=False,
        compute_backend="dask",
        reduce_mode="tree",
        reduce_n_batch=2,
    )
    from xradio.image import load_image

    persisted_image = load_image(str(image_store))
    validate_persisted_products(result["image"], persisted_image)
    export_products(
        {"image": persisted_image},
        args.output_dir / f"radps_{case_name}_{mode}{suffix}{clean_suffix}.npz",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=HARNESS_ROOT / "data" / "medium_tests"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=HARNESS_ROOT / "results" / "medium_tests" / "work",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HARNESS_ROOT / "results" / "medium_tests",
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["phase_center"])
    parser.add_argument(
        "--modes", nargs="+", choices=("mfs", "mvc"), default=["mfs", "mvc"]
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        choices=WEIGHTING_CASES,
        default=["natural_local"],
    )
    parser.add_argument("--n-chunks", type=int, default=2)
    parser.add_argument("--overwrite-conversion", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run the shared 0.1 Jy threshold-cleaning regression.",
    )
    parser.add_argument(
        "--astroviper-source",
        type=Path,
        default=None,
        help="Optional source tree to prepend; normally use an editable install.",
    )
    args = parser.parse_args()
    if args.astroviper_source is not None:
        sys.path.insert(0, str(args.astroviper_source))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Development source trees may be read-only to workers. Give Numba an
    # explicit writable cache instead of requiring an in-tree __pycache__.
    numba_cache = args.work_dir / "numba_cache"
    dask_cache = args.work_dir / "dask_cache"
    numba_cache.mkdir(parents=True, exist_ok=True)
    dask_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    from toolviper.dask.client import local_client

    client = local_client(
        cores=4,
        memory_limit="4GB",
        dask_local_dir=str(dask_cache),
    )
    try:
        for case_name, config in selected_cases(args.cases):
            for mode in args.modes:
                for weighting_name, weighting_config in selected_weightings(
                    args.weightings
                ):
                    run_case(
                        case_name,
                        config,
                        mode,
                        weighting_name,
                        weighting_config,
                        args,
                    )
    finally:
        client.close()


if __name__ == "__main__":
    main()
