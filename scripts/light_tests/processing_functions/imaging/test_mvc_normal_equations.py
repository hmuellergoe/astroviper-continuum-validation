"""Tests for MVC Taylor normal equations and primary-beam conventions."""

import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    apply_mvc_primary_beam_convention,
    convert_mvc_cubes_to_taylor_normal_equations,
)


def _cube(values, frequency):
    return xr.DataArray(
        np.asarray(values, dtype=np.float64)[None, :, None, None, None],
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            "time": [0],
            "frequency": frequency,
            "polarization": ["I"],
            "l": [0.0],
            "m": [0.0],
        },
    )


def _normalization(values, frequency):
    return xr.DataArray(
        np.asarray(values, dtype=np.float64)[None, :, None],
        dims=("time", "frequency", "polarization"),
        coords={
            "time": [0],
            "frequency": frequency,
            "polarization": ["I"],
        },
    )


def test_convert_mvc_cubes_forms_weighted_normal_equations():
    """MVC channel cubes reduce to the expected weighted Taylor equations."""
    frequency = np.array([0.8e9, 1.0e9, 1.2e9])
    spectral_coordinate = np.array([-0.2, 0.0, 0.2])
    residual = np.array([2.0, 4.0, 8.0])
    psf = np.array([1.0, 2.0, 4.0])
    primary_beam = np.array([0.5, 1.0, 1.5])
    weight = np.array([1.0, 2.0, 3.0])

    residual_taylor, psf_taylor, effective_pb = (
        convert_mvc_cubes_to_taylor_normal_equations(
            _cube(residual, frequency),
            _cube(psf, frequency),
            _cube(primary_beam, frequency),
            _normalization(weight, frequency),
            _normalization(weight, frequency),
            nterms=2,
            reference_frequency=1.0e9,
            pblimit=0.2,
        )
    )

    expected_effective_pb = np.sum(weight * primary_beam) / np.sum(weight)
    expected_residual = np.array(
        [
            np.sum(weight * residual) / np.sum(weight),
            np.sum(weight * spectral_coordinate * residual) / np.sum(weight),
        ]
    )
    expected_psf = np.array(
        [
            np.sum(weight * spectral_coordinate**order * primary_beam * psf)
            / (np.sum(weight) * expected_effective_pb)
            for order in range(3)
        ]
    )

    np.testing.assert_allclose(effective_pb.values.squeeze(), expected_effective_pb)
    np.testing.assert_allclose(residual_taylor.values.squeeze(), expected_residual)
    np.testing.assert_allclose(psf_taylor.values.squeeze(), expected_psf)
    assert residual_taylor.sizes["taylor_term"] == 2
    assert psf_taylor.sizes["psf_taylor_order"] == 3


def test_convert_mvc_cubes_masks_channels_below_primary_beam_limit():
    """MVC excludes channel contributions below the primary-beam cutoff."""
    frequency = np.array([0.9e9, 1.1e9])
    residual_taylor, _, _ = convert_mvc_cubes_to_taylor_normal_equations(
        _cube([5.0, 100.0], frequency),
        _cube([1.0, 1.0], frequency),
        _cube([1.0, 0.1], frequency),
        _normalization([1.0, 1.0], frequency),
        _normalization([1.0, 1.0], frequency),
        nterms=1,
        reference_frequency=1.0e9,
        pblimit=0.2,
    )

    np.testing.assert_allclose(residual_taylor.values.squeeze(), 2.5)


def test_apply_mvc_primary_beam_convention_uses_channel_over_common_beam():
    """MVC prediction scales its common-beam model by each channel PB."""
    frequency = np.array([0.9e9, 1.1e9])
    model = _cube([10.0, 20.0], frequency)
    channel_pb = _cube([0.5, 1.5], frequency)
    channel_pb = channel_pb.assign_coords(polarization=["XX"])
    effective_pb = xr.DataArray(
        np.array([[[[1.0]]]]),
        dims=("time", "polarization", "l", "m"),
        coords={"time": [0], "polarization": ["I"], "l": [0.0], "m": [0.0]},
    )

    actual = apply_mvc_primary_beam_convention(model, channel_pb, effective_pb)

    np.testing.assert_allclose(actual.values.squeeze(), [5.0, 30.0])
    assert actual.attrs["primary_beam_convention"] == "channel_pb_over_effective_pb"
