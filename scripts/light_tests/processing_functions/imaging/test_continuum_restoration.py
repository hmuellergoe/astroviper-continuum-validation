"""Tests for final continuum restoration and primary-beam correction."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    restore_image,
)


def _restored_image(primary_beam):
    """Create a restored Taylor stack and reference-frequency primary beam."""
    n_frequency = np.asarray(primary_beam).size // 4
    residual = np.stack(
        [np.full((1, 1, 2, 2), 4.0), np.full((1, 1, 2, 2), 40.0)],
        axis=1,
    )
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                residual,
            ),
            "SKY_MODEL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.zeros_like(residual),
            ),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                np.zeros((1, 3, 1, 2, 2)),
            ),
            "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION": (
                ("time", "frequency", "polarization", "beam_params"),
                np.tile(
                    np.asarray([1.0, 1.0, 0.0]),
                    (1, n_frequency, 1, 1),
                ),
            ),
            "PRIMARY_BEAM": (
                ("time", "frequency", "polarization", "l", "m"),
                np.asarray(primary_beam, dtype=np.float64).reshape(1, -1, 1, 2, 2),
            ),
        },
        coords={
            "time": [0],
            "taylor_term": [0, 1],
            "psf_taylor_order": [0, 1, 2],
            "frequency": np.arange(n_frequency),
            "polarization": ["I"],
            "beam_params": ["major", "minor", "pa"],
            "l": [0.0, 1.0],
            "m": [0.0, 1.0],
        },
        attrs={
            "type": "image_dataset",
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                    "beam_fit_params_point_spread_function": (
                        "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"
                    ),
                    "primary_beam": "PRIMARY_BEAM",
                },
                "model": {"sky": "SKY_MODEL"},
            },
        },
    )
    return image


def test_pbcor_returns_only_restored_taylor_zero_and_masks_below_limit():
    """PB correction produces one reference-frequency plane from Taylor zero."""
    image = _restored_image([1.0, 0.5, 0.1, np.nan])

    result, _ = restore_image(
        image,
        primary_beam_correction=True,
        primary_beam_limit=0.2,
    )
    corrected = result["SKY_RESTORED_PBCOR"]

    assert corrected.dims == ("time", "polarization", "l", "m")
    np.testing.assert_allclose(
        corrected.values,
        [[[[4.0, 8.0], [np.nan, np.nan]]]],
        equal_nan=True,
    )
    assert result.attrs["data_groups"]["restored_pbcor"]["sky"] == (
        "SKY_RESTORED_PBCOR"
    )


def test_pbcor_rejects_a_channelized_primary_beam_cube():
    """PB correction rejects a PB cube that was not reduced to one frequency."""
    image = _restored_image(np.ones(8))

    with pytest.raises(ValueError, match="one averaged primary-beam"):
        restore_image(image, primary_beam_correction=True)
