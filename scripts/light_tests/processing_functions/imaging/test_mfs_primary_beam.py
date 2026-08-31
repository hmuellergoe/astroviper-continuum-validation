"""Tests for construction of the MFS reference-frequency primary beam."""

import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _convert_primary_beam_to_reference_frequency,
)
from astroviper.processing_functions.imaging.primary_beam.make_pb_symmetric import (
    airy_disk_rorder_v2,
)


def test_mfs_primary_beam_is_evaluated_at_reference_frequency():
    """MFS replaces channel PB accumulators with a reference-frequency PB."""
    image_params = {
        "image_size": [8, 8],
        "cell_size": np.asarray([-2.0e-4, 2.0e-4]),
        "list_dish_diameters": np.asarray([10.7]),
        "list_blockage_diameters": np.asarray([0.75]),
        "primary_beam_ipower": 2,
    }
    image = xr.Dataset(
        coords={
            "time": [0],
            "frequency": [1.0e9, 2.0e9],
            "polarization": ["XX"],
            "l": np.arange(8),
            "m": np.arange(8),
        }
    )
    image.attrs["data_groups"] = {"residual": {}}

    result = _convert_primary_beam_to_reference_frequency(
        image,
        image_params=image_params,
        reference_frequency_hz=1.5e9,
        float_dtype=np.float64,
    )

    expected = airy_disk_rorder_v2(
        np.asarray([1.5e9]),
        np.asarray(["XX"]),
        {
            "list_dish_diameters": np.asarray([10.7]),
            "list_blockage_diameters": np.asarray([0.75]),
            "ipower": 2,
        },
        {**image_params, "image_center": [4, 4]},
        dtype=np.float64,
    )[0, 0, ...][None, ...]

    assert "PRIMARY_BEAM" not in result
    assert "PRIMARY_BEAM_SUM" not in result
    assert "PRIMARY_BEAM_CHANNEL_COUNT" not in result
    np.testing.assert_allclose(result["PRIMARY_BEAM_REFERENCE"], expected)
    assert result["PRIMARY_BEAM_REFERENCE"].attrs["reference_frequency_hz"] == 1.5e9
    assert result.attrs["data_groups"]["residual"] == {
        "primary_beam_reference": "PRIMARY_BEAM_REFERENCE"
    }
