"""Tests for final continuum restoration and primary-beam correction."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    primary_beam_correct_restored_continuum,
)


def _restored_image(primary_beam):
    """Create a restored Taylor stack and reference-frequency primary beam."""
    restored = np.stack(
        [np.full((1, 1, 2, 2), 4.0), np.full((1, 1, 2, 2), 40.0)],
        axis=1,
    )
    image = xr.Dataset(
        {
            "SKY_RESTORED": (
                ("time", "taylor_term", "polarization", "l", "m"),
                restored,
            ),
            "PRIMARY_BEAM": (
                ("time", "frequency", "polarization", "l", "m"),
                np.asarray(primary_beam, dtype=np.float64).reshape(1, -1, 1, 2, 2),
            ),
        },
        coords={
            "time": [0],
            "taylor_term": [0, 1],
            "frequency": np.arange(np.asarray(primary_beam).size // 4),
            "polarization": ["I"],
            "l": [0.0, 1.0],
            "m": [0.0, 1.0],
        },
        attrs={
            "type": "image_dataset",
            "data_groups": {"restored": {"sky": "SKY_RESTORED"}},
        },
    )
    return image


def test_pbcor_returns_only_restored_taylor_zero_and_masks_below_limit():
    """PB correction produces one reference-frequency plane from Taylor zero."""
    image = _restored_image([1.0, 0.5, 0.1, np.nan])

    result = primary_beam_correct_restored_continuum(image, pblimit=0.2)
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

    with pytest.raises(ValueError, match="one averaged primary beam"):
        primary_beam_correct_restored_continuum(image)
