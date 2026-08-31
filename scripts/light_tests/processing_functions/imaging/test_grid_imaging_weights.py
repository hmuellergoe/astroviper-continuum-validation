"""Tests for continuum-compatible imaging-weight grid cell conventions."""

import numpy as np

from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights import (
    degrid_imaging_weights,
    grid_imaging_weights,
)


def _grid(truncate_uv_cells):
    speed_of_light = 299792458.0
    grid = np.zeros((1, 1, 8, 8), dtype=np.float64)
    sum_weight = np.zeros((1, 1), dtype=np.float64)
    uvw = np.asarray([[[-0.75, 0.0, 0.0]]], dtype=np.float64)
    data_weight = np.asarray([[[[2.0]]]], dtype=np.float64)
    frequency = np.asarray([speed_of_light], dtype=np.float64)
    grid_imaging_weights(
        grid,
        sum_weight,
        uvw,
        data_weight,
        frequency,
        np.asarray([8, 8]),
        np.asarray([0.1, 0.1]),
        truncate_uv_cells=truncate_uv_cells,
    )
    return grid, sum_weight, uvw, data_weight, frequency


def test_continuum_cell_assignment_truncates_instead_of_rounding():
    """Continuum density gridding can use CASA-compatible cell truncation."""
    nearest, _, _, _, _ = _grid(False)
    truncated, sum_weight, _, _, _ = _grid(True)

    assert nearest[0, 0, 5, 4] == 2.0
    assert truncated[0, 0, 4, 4] == 2.0
    assert nearest[0, 0, 4, 4] == 0.0
    np.testing.assert_allclose(sum_weight, [[4.0]])


def test_continuum_degrid_samples_the_truncated_density_cell():
    """Briggs degridding samples the same truncated cell used during gridding."""
    grid, _, uvw, data_weight, frequency = _grid(True)
    briggs_factors = np.asarray([[[0.5]], [[1.0]]], dtype=np.float64)

    result = degrid_imaging_weights(
        grid,
        uvw,
        data_weight,
        briggs_factors,
        frequency,
        np.asarray([8, 8]),
        np.asarray([0.1, 0.1]),
        truncate_uv_cells=True,
    )

    # 2 / (0.5 * density(4, 4) + 1) = 2 / 2 = 1.
    np.testing.assert_allclose(result, 1.0)
