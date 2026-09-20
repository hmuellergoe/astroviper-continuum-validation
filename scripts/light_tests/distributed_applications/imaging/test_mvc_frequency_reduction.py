"""Tests for combining distributed MFS and MVC map products."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    combine_continuum_chunks,
)


def _mvc_map_result(value, task_id):
    residual = xr.DataArray(
        np.full((1, 2, 1, 2, 2), float(value)),
        dims=("time", "taylor_term", "polarization", "l", "m"),
        coords={
            "time": [0],
            "taylor_term": [0, 1],
            "polarization": ["XX"],
            "l": [0.0, 1.0],
            "m": [0.0, 1.0],
        },
    )
    image = xr.Dataset(
        {
            "MVC_RESIDUAL_TAYLOR_NUMERATOR": residual,
            "MVC_RESIDUAL_WEIGHT_SUM": xr.DataArray(
                np.full((1, 1), float(value)),
                dims=("time", "polarization"),
            ),
        }
    )
    image.attrs["continuum_imaging"] = {
        "specmode": "mvc",
        "nterms": 2,
        "reference_frequency_hz": 1.0e9,
    }
    return {
        "image": image,
        "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
    }


def _reduce(results):
    return combine_continuum_chunks(
        results,
        {
            "specmode": "mvc",
            "additive_variables": (
                "MVC_RESIDUAL_TAYLOR_NUMERATOR",
                "MVC_RESIDUAL_WEIGHT_SUM",
            ),
        },
    )


def test_mvc_reducer_sums_map_local_taylor_contributions():
    """MVC reduction adds compact Taylor numerators and normalization sums."""
    result = _reduce(
        [
            _mvc_map_result(1.0, 1),
            _mvc_map_result(2.0, 2),
        ]
    )

    np.testing.assert_allclose(result["image"]["MVC_RESIDUAL_TAYLOR_NUMERATOR"], 3.0)
    np.testing.assert_allclose(result["image"]["MVC_RESIDUAL_WEIGHT_SUM"], 3.0)
    assert "frequency" not in result["image"].dims


def test_mvc_reducer_preserves_task_local_visibility_grid_caches():
    """Tree reduction keeps every MVC task's observed UV grid separately."""
    first = _mvc_map_result(1.0, 1)
    second = _mvc_map_result(2.0, 2)
    first["task_id"] = 1
    second["task_id"] = 2
    first["observed_visibility_grid_xds"] = xr.Dataset(
        {"VISIBILITY": xr.DataArray([1.0 + 0.0j], dims=("frequency",))}
    )
    second["observed_visibility_grid_xds"] = xr.Dataset(
        {"VISIBILITY": xr.DataArray([2.0 + 0.0j], dims=("frequency",))}
    )

    result = _reduce([first, second])

    assert set(result["observed_visibility_grid_mapping"]) == {1, 2}
    np.testing.assert_array_equal(
        result["observed_visibility_grid_mapping"][1].VISIBILITY,
        [1.0 + 0.0j],
    )
    np.testing.assert_array_equal(
        result["observed_visibility_grid_mapping"][2].VISIBILITY,
        [2.0 + 0.0j],
    )


def _mfs_map_result(primary_beam, task_id):
    image = xr.Dataset(
        {
            "PRIMARY_BEAM_REFERENCE": xr.DataArray(
                np.asarray(primary_beam, dtype=np.float64).reshape(1, 1, 2, 2),
                dims=("time", "polarization", "l", "m"),
            )
        }
    )
    image.attrs["continuum_imaging"] = {
        "specmode": "mfs",
        "nterms": 2,
        "reference_frequency_hz": 1.5e9,
    }
    return {
        "image": image,
        "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
    }


def test_mfs_reducer_keeps_identical_reference_primary_beam_once():
    """MFS reduction retains one copy of an identical static reference PB."""
    beam = np.arange(4, dtype=np.float64)
    result = combine_continuum_chunks(
        [_mfs_map_result(beam, 1), _mfs_map_result(beam, 2)],
        {"specmode": "mfs", "additive_variables": ()},
    )

    np.testing.assert_array_equal(
        result["image"]["PRIMARY_BEAM_REFERENCE"].values.reshape(-1), beam
    )


def test_mfs_reducer_keeps_the_available_reference_primary_beam():
    """MFS reduction accepts a static PB supplied by only one input leaf."""
    beam = np.arange(4, dtype=np.float64)
    without_beam = _mfs_map_result(beam, 1)
    without_beam["image"] = without_beam["image"].drop_vars("PRIMARY_BEAM_REFERENCE")
    result = combine_continuum_chunks(
        [without_beam, _mfs_map_result(beam, 2)],
        {"specmode": "mfs", "additive_variables": ()},
    )

    np.testing.assert_array_equal(
        result["image"]["PRIMARY_BEAM_REFERENCE"].values.reshape(-1), beam
    )


def test_mfs_reducer_rejects_inconsistent_reference_primary_beam():
    """MFS reduction rejects inconsistent copies of a static reference PB."""
    with pytest.raises(ValueError, match="Static variable.*differs"):
        combine_continuum_chunks(
            [
                _mfs_map_result(np.ones(4), 1),
                _mfs_map_result(np.full(4, 2.0), 2),
            ],
            {"specmode": "mfs", "additive_variables": ()},
        )
