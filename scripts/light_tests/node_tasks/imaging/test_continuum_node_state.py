"""Tests for continuum node-task static products and cached graph state."""

import numpy as np
import pytest
import xarray as xr

import astroviper.node_tasks.imaging.image_continuum_single_field as continuum_node


def _static_products():
    return xr.Dataset(
        {
            "POINT_SPREAD_FUNCTION": xr.DataArray(
                np.ones((1, 3, 1, 2, 2)),
                dims=("time", "psf_taylor_order", "polarization", "l", "m"),
            ),
            "PRIMARY_BEAM": xr.DataArray(
                np.ones((1, 1, 1, 2, 2)),
                dims=("time", "frequency", "polarization", "l", "m"),
            ),
            "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": xr.DataArray(
                np.ones((1, 1, 3)),
                dims=("time", "polarization", "beam_params_label"),
            ),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": xr.DataArray(
                np.ones((1, 1)),
                dims=("time", "polarization"),
            ),
        },
        coords={"frequency": [1.5e9]},
    )


def test_install_static_products_replaces_unused_chunk_frequency_coordinate():
    """A reduced reference-frequency coordinate replaces unused chunk channels."""
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": xr.DataArray(
                np.zeros((1, 2, 1, 2, 2)),
                dims=("time", "taylor_term", "polarization", "l", "m"),
            )
        },
        coords={"frequency": [1.0e9, 2.0e9]},
    )

    result = continuum_node._install_static_continuum_products(
        image,
        _static_products(),
    )

    np.testing.assert_array_equal(result.frequency.values, [1.5e9])
    assert result["PRIMARY_BEAM"].sizes["frequency"] == 1


def test_install_static_products_rejects_frequency_coordinate_still_in_use():
    """Static installation cannot replace frequency while a cube still uses it."""
    image = xr.Dataset(
        {
            "CHANNEL_DATA": xr.DataArray(
                np.zeros((2, 2, 2)),
                dims=("frequency", "l", "m"),
            )
        },
        coords={"frequency": [1.0e9, 2.0e9]},
    )

    with pytest.raises(ValueError, match="variables still use it"):
        continuum_node._install_static_continuum_products(
            image,
            _static_products(),
        )


def test_single_leaf_state_is_normalized_to_cache_mappings(monkeypatch):
    """A one-leaf reduce result is normalized to task-keyed cache mappings."""
    pb_xds = xr.Dataset({"PRIMARY_BEAM": xr.DataArray([1.0], dims=("frequency",))})
    weight_datasets = {"ms_0": xr.Dataset({"WEIGHT_IMAGING": xr.DataArray([1.0])})}
    observed_grid_mapping = {7: xr.Dataset({"VISIBILITY": xr.DataArray([2.0 + 0.0j])})}
    captured = {}

    def fake_prepare(image, input_params, **kwargs):
        captured["pb_cache_mapping"] = kwargs["pb_cache_mapping"]
        return image, xr.Dataset(), None

    monkeypatch.setattr(continuum_node, "_prepare_continuum_image", fake_prepare)
    monkeypatch.setattr(
        continuum_node,
        "model_update_continuum_single_field",
        lambda input_data, input_params: {"image": input_data["image"]},
    )
    monkeypatch.setattr(
        continuum_node,
        "_prepare_post_update_continuum_model_state",
        lambda image, input_params: (xr.Dataset(), None),
    )

    result = continuum_node.continuum_minor_cycle_node(
        {
            "image": xr.Dataset(),
            "task_id": 7,
            "pb_xds": pb_xds,
            "weight_datasets": weight_datasets,
            "observed_visibility_grid_mapping": observed_grid_mapping,
        },
        {"is_n_iter_0": True},
    )

    assert captured["pb_cache_mapping"] == {7: pb_xds}
    assert result["pb_cache_mapping"] == {7: pb_xds}
    assert result["weight_cache_mapping"] == {7: weight_datasets}
    assert result["observed_visibility_grid_mapping"] is observed_grid_mapping
    assert isinstance(result["model_xds"], xr.Dataset)
    assert result["model_uv_xds"] is None
