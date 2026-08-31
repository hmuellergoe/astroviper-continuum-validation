"""Guard continuum development against changes to established cube imaging.

The reference is the five-channel TW Hydra dirty cube produced by the canonical
AstroVIPER cube tutorial setup.  This test deliberately exercises the public
distributed application and reopens its persisted Zarr output before comparing
the cube products channel by channel.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging.image_cube_single_field import (
    image_cube_single_field,
)
from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    image_continuum_single_field,
)

HARNESS_ROOT = Path(__file__).resolve().parents[3]
PS_STORE = (
    HARNESS_ROOT
    / "data"
    / "medium_tests"
    / "cube_backward_compatibility"
    / "twhya_selfcal_lsrk_5chans.ps.zarr"
)
REFERENCE_STORE = (
    HARNESS_ROOT
    / "results"
    / "medium_tests"
    / "cube_backward_compatibility"
    / "twhya_dirty.img.zarr"
)
RELATIVE_TOLERANCE = 1e-6

IMAGING_WEIGHTS_PARAMS = {
    "weighting": "briggs",
    "robust": 0.5,
    "casa_weighting_implementation": True,
}

DIRTY_ITERATION_CONTROL = {
    "niter": 0,
    "nmajor": 0,
    "threshold": 0.0,
    "gain": 0.1,
    "cyclefactor": 1.5,
    "cycleniter": -1,
    "minpsffraction": 0.05,
    "maxpsffraction": 0.8,
}

IMAGE_VARIABLES = (
    "SKY_RESIDUAL",
    "POINT_SPREAD_FUNCTION",
    "PRIMARY_BEAM",
)

CONTINUUM_VARIABLES = (
    "SKY_RESIDUAL",
    "POINT_SPREAD_FUNCTION",
    "PRIMARY_BEAM",
)


def _image_params(processing_set):
    """Build the canonical TW Hydra cube geometry from its input coordinates."""
    field_xds = processing_set.xr_ps.get_combined_field_and_source_xds()
    center_field_name = field_xds.attrs["center_field_name"]
    phase_direction = field_xds.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=center_field_name
    )
    return {
        "image_size": [250, 250],
        "cell_size": np.array([-0.1, 0.1]) * np.pi / (180 * 3600),
        "phase_direction": phase_direction.values,
        "frequency_coords": processing_set.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }


def _continuum_image_params(processing_set):
    """Build an MFS geometry spanning all five TW Hydra channels."""
    image_params = _image_params(processing_set)
    frequency = np.asarray(image_params["frequency_coords"], dtype=np.float64)
    image_params.update(
        {
            "nterms": 2,
            "reference_frequency": float(np.mean(frequency)),
            "reference_frequency_hz": float(np.mean(frequency)),
        }
    )
    return image_params


def _assert_cube_product_unchanged(actual, reference, variable):
    """Compare every frequency/polarization plane at the cube truth tolerance."""
    assert actual[variable].dims == reference[variable].dims
    actual_values = np.asarray(actual[variable].values)
    reference_values = np.asarray(reference[variable].values)
    assert actual_values.shape == reference_values.shape
    assert np.array_equal(np.isfinite(actual_values), np.isfinite(reference_values))

    for frequency in range(reference.sizes["frequency"]):
        for polarization in range(reference.sizes["polarization"]):
            selection = {
                "time": 0,
                "frequency": frequency,
                "polarization": polarization,
            }
            actual_plane = np.asarray(actual[variable].isel(selection).values)
            reference_plane = np.asarray(reference[variable].isel(selection).values)
            denominator = np.nanmax(np.abs(reference_plane))
            difference = np.nanmax(np.abs(actual_plane - reference_plane))
            relative_difference = (
                difference / denominator if denominator else difference
            )
            assert relative_difference < RELATIVE_TOLERANCE, (
                f"{variable}, channel {frequency}, polarization {polarization}: "
                f"peak relative difference {relative_difference:.3e} exceeds "
                f"{RELATIVE_TOLERANCE:.1e}"
            )


def test_tw_hydra_dirty_cube_is_backward_compatible(tmp_path):
    """The public cube application must reproduce the persisted TW Hydra baseline."""
    processing_set = open_processing_set(str(PS_STORE))
    output_store = tmp_path / "twhya_dirty_current.img.zarr"

    with dask.config.set(scheduler="synchronous"):
        image_cube_single_field(
            ps_store=str(PS_STORE),
            image_store=str(output_store),
            image_params=_image_params(processing_set),
            imaging_weights_params=IMAGING_WEIGHTS_PARAMS,
            iteration_control_params=DIRTY_ITERATION_CONTROL,
            gridder="prolate_spheroidal",
            deconvolver="hogbom_many_threads",
            scan_intents="OBSERVE_TARGET#ON_SOURCE",
            image_data_variables_keep=[
                "sky_residual",
                "point_spread_function",
                "primary_beam",
                "beam_fit_params_point_spread_function",
            ],
            processing_set_data_group_name="base",
            single_precision_image=False,
            thread_info=None,
            processing_function_threads=1,
            n_chunks=5,
            overwrite=True,
            disk_chunk_sizes={"frequency": 2},
            vizualize_graph=False,
        )

    actual = xr.open_zarr(output_store)
    reference = xr.open_zarr(REFERENCE_STORE)

    assert actual.sizes == reference.sizes
    assert np.array_equal(
        actual.polarization.values,
        reference.polarization.values,
    )
    for coordinate in ("time", "frequency", "l", "m"):
        np.testing.assert_allclose(
            actual[coordinate].values,
            reference[coordinate].values,
            rtol=0.0,
            atol=0.0,
        )

    for variable in IMAGE_VARIABLES:
        _assert_cube_product_unchanged(actual, reference, variable)

    np.testing.assert_allclose(
        actual.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
        reference.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
        rtol=RELATIVE_TOLERANCE,
        atol=0.0,
    )


def _run_tw_hydra_continuum(
    output_store, processing_set, n_chunks, specmode, imaging_weights_params
):
    """Run the public distributed MFS application with fixed partitioning."""
    image_continuum_single_field(
        ps_store=str(PS_STORE),
        image_store=str(output_store),
        image_params=_continuum_image_params(processing_set),
        imaging_weights_params=imaging_weights_params,
        iteration_control_params={
            **DIRTY_ITERATION_CONTROL,
            # The current compatibility minor loop requires a positive budget;
            # the unreachable threshold keeps this a dirty-image comparison.
            "niter": 1,
            "cycleniter": 1,
            "threshold": 1.0e30,
        },
        gridder="prolate_spheroidal",
        deconvolver="hogbom",
        restore=False,
        pbcor=False,
        specmode=specmode,
        scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
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
        n_chunks=n_chunks,
        overwrite=True,
        memory_mode="in_memory",
        vizualize_graph=False,
        compute_backend="dask",
        reduce_mode="tree",
        reduce_n_batch=2,
    )
    return xr.open_zarr(output_store)


@pytest.mark.parametrize(
    ("specmode", "weighting_name", "imaging_weights_params"),
    [
        ("mfs", "natural_local", {"weighting": "natural", "weighting_scope": "local"}),
        ("mvc", "natural_local", {"weighting": "natural", "weighting_scope": "local"}),
        (
            "mfs",
            "briggs_global",
            {
                "weighting": "briggs",
                "robust": 0.5,
                "weighting_scope": "global",
                "casa_weighting_implementation": True,
            },
        ),
    ],
)
def test_tw_hydra_continuum_is_invariant_to_distributed_frequency_partitioning(
    tmp_path, specmode, weighting_name, imaging_weights_params
):
    """One and five chunks produce equivalent continuum Taylor products."""
    processing_set = open_processing_set(str(PS_STORE))
    one_chunk = _run_tw_hydra_continuum(
        tmp_path / f"twhya_{specmode}_{weighting_name}_one_chunk.img.zarr",
        processing_set,
        1,
        specmode,
        imaging_weights_params,
    )
    five_chunks = _run_tw_hydra_continuum(
        tmp_path / f"twhya_{specmode}_{weighting_name}_five_chunks.img.zarr",
        processing_set,
        5,
        specmode,
        imaging_weights_params,
    )

    assert five_chunks.SKY_RESIDUAL.sizes["taylor_term"] == 2
    assert five_chunks.POINT_SPREAD_FUNCTION.sizes["psf_taylor_order"] == 3
    assert five_chunks.PRIMARY_BEAM.sizes["frequency"] == 1

    for variable in CONTINUUM_VARIABLES:
        actual = np.asarray(five_chunks[variable].values)
        reference = np.asarray(one_chunk[variable].values)
        assert actual.shape == reference.shape
        assert np.array_equal(np.isfinite(actual), np.isfinite(reference))
        denominator = np.nanmax(np.abs(reference))
        difference = np.nanmax(np.abs(actual - reference))
        relative_difference = difference / denominator if denominator else difference
        assert relative_difference < RELATIVE_TOLERANCE, (
            f"{specmode}/{weighting_name}/{variable}: peak relative difference "
            f"{relative_difference:.3e} exceeds {RELATIVE_TOLERANCE:.1e}"
        )


if __name__ == "__main__":
    with TemporaryDirectory(prefix="tw-hydra-cube-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        test_tw_hydra_dirty_cube_is_backward_compatible(temporary_path)
        for specmode, weighting_name, weighting_params in (
            (
                "mfs",
                "natural_local",
                {"weighting": "natural", "weighting_scope": "local"},
            ),
            (
                "mvc",
                "natural_local",
                {"weighting": "natural", "weighting_scope": "local"},
            ),
            (
                "mfs",
                "briggs_global",
                {
                    "weighting": "briggs",
                    "robust": 0.5,
                    "weighting_scope": "global",
                    "casa_weighting_implementation": True,
                },
            ),
        ):
            test_tw_hydra_continuum_is_invariant_to_distributed_frequency_partitioning(
                temporary_path, specmode, weighting_name, weighting_params
            )
    print("TW Hydra cube and continuum distributed tests: PASS")
