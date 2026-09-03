#!/usr/bin/env python3
"""Evaluate continuum science metrics and ALMA QA2 image/data checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
OUTPUT_ROOT = ROOT / "evaluation"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/astroviper-continuum-evaluation-matplotlib")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAD_TO_ARCSEC = 180.0 * 3600.0 / np.pi
GAUSSIAN_BEAM_FACTOR = np.pi / (4.0 * np.log(2.0))
SPEED_OF_LIGHT_M_S = 299_792_458.0
QA2_NOISE_TOLERANCE = {
    **{band: 0.10 for band in range(1, 7)},
    **{band: 0.15 for band in (7, 8)},
    **{band: 0.20 for band in (9, 10)},
}
QA2_FLUX_TOLERANCE = {
    **{band: 0.05 for band in range(1, 6)},
    **{band: 0.10 for band in (6, 7)},
    **{band: 0.20 for band in (8, 9, 10)},
}
ALMA_BAND_RANGES_GHZ = {
    1: (35.0, 50.0),
    2: (67.0, 90.0),
    3: (84.0, 116.0),
    4: (125.0, 163.0),
    5: (163.0, 211.0),
    6: (211.0, 275.0),
    7: (275.0, 373.0),
    8: (385.0, 500.0),
    9: (602.0, 720.0),
    10: (787.0, 950.0),
}


def _stokes_i_plane(data_array, **indices):
    """Select time zero, Stokes I, and requested scalar dimensions."""
    selection = {"time": 0, **indices}
    if "polarization" in data_array.dims:
        data_array = data_array.sel(polarization="I")
    return np.asarray(data_array.isel(selection).values).squeeze()


def _robust_sigma(values):
    """Return Gaussian-equivalent sigma from the median absolute deviation."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def _maximum_absolute(values):
    """Return the largest finite absolute value, or NaN for an empty selection."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.max(np.abs(values))) if values.size else np.nan


def _infer_alma_band(frequency_hz):
    """Infer an unambiguous ALMA receiver band from one frequency."""
    frequency_ghz = frequency_hz / 1.0e9
    matches = [
        band
        for band, limits in ALMA_BAND_RANGES_GHZ.items()
        if limits[0] <= frequency_ghz <= limits[1]
    ]
    return matches[0] if len(matches) == 1 else None


def _range_status(value, lower, upper):
    """Assess a value against optional inclusive limits."""
    if lower is None and upper is None:
        return "NOT_ASSESSABLE"
    if lower is not None and value < lower:
        return "FAIL"
    if upper is not None and value > upper:
        return "FAIL"
    return "PASS"


def _noise_status(noise, goal, tolerance):
    """Apply the QA2 upper noise allowance; better-than-goal noise passes."""
    if goal is None or tolerance is None:
        return "NOT_ASSESSABLE"
    return "PASS" if noise <= goal * (1.0 + tolerance) else "FAIL"


def _flux_status(flux, reference_flux, tolerance):
    """Assess absolute flux recovery when an external reference is supplied."""
    if reference_flux is None or tolerance is None:
        return "NOT_ASSESSABLE"
    if reference_flux == 0.0:
        raise ValueError("Reference flux must be nonzero.")
    relative_error = abs(flux / reference_flux - 1.0)
    within_limit = relative_error <= tolerance or np.isclose(
        relative_error, tolerance, rtol=1.0e-12, atol=0.0
    )
    return "PASS" if within_limit else "FAIL"


def _qa2_goals(arguments, observation):
    """Normalize CLI goals and attach band-dependent QA2 tolerances."""
    band = arguments.alma_band or _infer_alma_band(
        observation["reference_frequency_hz"]
    )
    return {
        "alma_band": band,
        "alma_band_source": "command line" if arguments.alma_band else "frequency",
        "noise_goal_jy_per_beam": arguments.noise_goal_jy_per_beam,
        "requested_bandwidth_hz": arguments.requested_bandwidth_hz,
        "representative_frequency_hz": arguments.representative_frequency_hz,
        "resolution_min_arcsec": arguments.resolution_min_arcsec,
        "resolution_max_arcsec": arguments.resolution_max_arcsec,
        "beam_area_min_arcsec2": arguments.beam_area_min_arcsec2,
        "beam_area_max_arcsec2": arguments.beam_area_max_arcsec2,
        "reference_flux_jy": arguments.reference_flux_jy,
        "noise_tolerance_fraction": QA2_NOISE_TOLERANCE.get(band),
        "flux_tolerance_fraction": QA2_FLUX_TOLERANCE.get(band),
    }


def _configuration_paths(project, variant):
    """Return completed configuration directories for one source variant."""
    root = RESULTS_ROOT / project / variant / "astroviper"
    paths = sorted(path.parent for path in root.glob("*/complete.json"))
    if not paths:
        raise FileNotFoundError(f"No completed configurations found below {root}")
    return paths


def _completed_variants():
    """Return project/variant pairs containing at least one completed result."""
    variants = {
        (manifest.parents[3].name, manifest.parents[2].name)
        for manifest in RESULTS_ROOT.glob("*/*/astroviper/*/complete.json")
    }
    return sorted(variants)


def _manifest(path):
    """Read one result manifest."""
    return json.loads((path / "complete.json").read_text())


def _experiment_identity(path):
    """Return normalized experiment metadata independently of directory naming."""
    experiment = _manifest(path).get("experiment", {})
    return {
        "name": experiment.get("name", path.name),
        "specmode": experiment.get("specmode"),
        "weighting_scope": experiment.get("weighting_scope"),
        "weight_implementation": experiment.get("weight_implementation"),
        "use_casa_weights": experiment.get("use_casa_weights"),
    }


def _reference_path(configuration_paths, requested=None):
    """Resolve a directory or manifest experiment name to one result path."""
    by_directory = {path.name: path for path in configuration_paths}
    if requested in by_directory:
        return by_directory[requested]

    identities = {path: _experiment_identity(path) for path in configuration_paths}
    if requested is not None:
        matches = [
            path
            for path, identity in identities.items()
            if identity["name"] == requested
        ]
        if not matches:
            available = sorted(
                set(by_directory)
                | {identity["name"] for identity in identities.values()}
            )
            raise KeyError(
                f"Reference {requested!r} is unavailable; choose from {available}"
            )
        return sorted(matches, key=lambda path: ("__" in path.name, path.name))[0]

    priority = (
        ("mfs", "global", True),
        ("mfs", "global", False),
        ("mfs", "local", True),
        ("mvc", "global", True),
    )
    for specmode, scope, use_casa_weights in priority:
        matches = [
            path
            for path, identity in identities.items()
            if identity["specmode"] == specmode
            and identity["weighting_scope"] == scope
            and identity["use_casa_weights"] is use_casa_weights
        ]
        if matches:
            return sorted(matches, key=lambda path: ("__" in path.name, path.name))[0]
    return sorted(configuration_paths)[0]


def _channel_width_hz(dataset):
    """Read the XRADIO scalar channel-width measure."""
    measure = dataset.frequency.attrs.get("channel_width", {})
    value = measure.get("data") if isinstance(measure, dict) else measure
    if value is None:
        frequencies = np.sort(np.asarray(dataset.frequency.values))
        if frequencies.size < 2:
            return np.nan
        value = np.median(np.diff(frequencies))
    return abs(float(np.asarray(value).squeeze()))


def _observation_metadata(configuration_paths):
    """Measure QA2 coverage quantities from the selected Processing Set."""
    manifest = json.loads((configuration_paths[0] / "complete.json").read_text())
    processing_set = Path(manifest["processing_set"])
    partitions = sorted(
        path
        for path in processing_set.iterdir()
        if path.is_dir() and (path / "zarr.json").exists()
    )
    if not partitions:
        raise FileNotFoundError(
            f"No Processing Set partitions found in {processing_set}"
        )

    frequency_ranges = []
    channel_widths = []
    frequency_sum = 0.0
    channel_count = 0
    visibility_count = 0
    flagged_count = 0
    time_unflagged = {}
    time_integration = {}
    first_dataset = None

    for partition in partitions:
        dataset = xr.open_zarr(partition)
        if first_dataset is None:
            first_dataset = dataset
        frequencies = np.asarray(dataset.frequency.values, dtype=np.float64)
        channel_width = _channel_width_hz(dataset)
        channel_widths.append(channel_width)
        frequency_ranges.append(
            {
                "partition": partition.name,
                "minimum_hz": float(np.min(frequencies) - channel_width / 2.0),
                "maximum_hz": float(np.max(frequencies) + channel_width / 2.0),
                "channel_width_hz": channel_width,
                "n_channels": int(frequencies.size),
            }
        )
        frequency_sum += float(np.sum(frequencies))
        channel_count += int(frequencies.size)

        flags = dataset.FLAG
        visibility_count += int(flags.size)
        flagged_count += int(flags.sum().compute().item())
        any_unflagged = np.asarray(
            (~flags.all(dim=("baseline_id", "frequency", "polarization")))
            .compute()
            .values
        )
        integration = np.asarray(dataset.EFFECTIVE_INTEGRATION_TIME.values)
        positive_integration = np.where(integration > 0.0, integration, np.nan)
        integration_per_time = np.nanmedian(positive_integration, axis=1)
        for time, duration, is_unflagged in zip(
            np.asarray(dataset.time.values),
            integration_per_time,
            any_unflagged,
            strict=True,
        ):
            key = float(time)
            if np.isfinite(duration):
                time_integration[key] = max(
                    time_integration.get(key, 0.0), float(duration)
                )
            time_unflagged[key] = time_unflagged.get(key, False) or bool(is_unflagged)

    uvw = np.asarray(first_dataset.UVW.values)
    uv_radius_m = np.hypot(uvw[..., 0], uvw[..., 1])
    uv_radius_m = uv_radius_m[np.isfinite(uv_radius_m) & (uv_radius_m > 0.0)]
    reference_frequency_hz = frequency_sum / channel_count
    wavelength_m = SPEED_OF_LIGHT_M_S / reference_frequency_hz
    uv_percentiles_m = np.percentile(uv_radius_m, [5.0, 50.0, 95.0])
    unflagged_time_seconds = sum(
        duration
        for time, duration in time_integration.items()
        if time_unflagged.get(time, False)
    )
    prepared_manifest = (
        RESULTS_ROOT
        / manifest["project"]
        / manifest["variant"]
        / "prepared"
        / "complete.json"
    )
    prepared = (
        json.loads(prepared_manifest.read_text()) if prepared_manifest.exists() else {}
    )
    source_measurement_sets = prepared.get("source_measurement_sets", [])
    polarization_products = [str(value) for value in first_dataset.polarization.values]

    return {
        "processing_set": str(processing_set),
        "source_measurement_sets": source_measurement_sets,
        "n_execution_blocks": len(source_measurement_sets),
        "field_selection": prepared.get("selection", {}).get("field"),
        "spw_selection": prepared.get("selection", {}).get("spw"),
        "source_datacolumn": prepared.get("source_datacolumn", "UNKNOWN"),
        "n_spectral_windows": len(partitions),
        "n_channels": channel_count,
        "frequency_ranges": frequency_ranges,
        "frequency_min_hz": min(item["minimum_hz"] for item in frequency_ranges),
        "frequency_max_hz": max(item["maximum_hz"] for item in frequency_ranges),
        "reference_frequency_hz": reference_frequency_hz,
        "channel_width_min_hz": float(np.min(channel_widths)),
        "channel_width_max_hz": float(np.max(channel_widths)),
        "aggregate_bandwidth_hz": float(
            sum(
                item["channel_width_hz"] * item["n_channels"]
                for item in frequency_ranges
            )
        ),
        "polarization_products": polarization_products,
        "full_polarization": {"XX", "XY", "YX", "YY"}.issubset(polarization_products),
        "visibility_samples": visibility_count,
        "flagged_visibility_samples": flagged_count,
        "flag_fraction": flagged_count / visibility_count,
        "on_source_time_seconds": float(sum(time_integration.values())),
        "unflagged_on_source_time_seconds": float(unflagged_time_seconds),
        "uv_radius_p05_m": float(uv_percentiles_m[0]),
        "uv_radius_median_m": float(uv_percentiles_m[1]),
        "uv_radius_p95_m": float(uv_percentiles_m[2]),
        "uv_radius_p95_klambda": float(uv_percentiles_m[2] / wavelength_m / 1.0e3),
        "largest_angular_scale_estimate_arcsec": float(
            0.6 * wavelength_m / uv_percentiles_m[0] * RAD_TO_ARCSEC
        ),
        "amplitude_scale_consistency_status": (
            "NOT_APPLICABLE" if len(source_measurement_sets) < 2 else "NOT_ASSESSABLE"
        ),
        "parallactic_angle_status": (
            "NOT_APPLICABLE"
            if not {"XY", "YX"}.intersection(polarization_products)
            else "NOT_ASSESSABLE"
        ),
        "renormalization_status": "NOT_RECORDED",
        "calibration_qa_status": "NOT_ASSESSABLE_FROM_IMAGE_PRODUCTS",
    }


def _geometry(configuration_paths, reference_name, aperture_beams, noise_beams):
    """Define one source center and common aperture/noise exclusion radii."""
    datasets = {
        path.name: xr.open_zarr(path / "continuum.img.zarr")
        for path in configuration_paths
    }
    if reference_name not in datasets:
        raise KeyError(
            f"Reference {reference_name!r} is unavailable; "
            f"choose from {sorted(datasets)}"
        )

    maximum_beam_fwhm_pixels = 0.0
    for dataset in datasets.values():
        beam = _stokes_i_plane(dataset.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION)
        pixel_scale = np.sqrt(
            abs(float(np.diff(dataset.l.values[:2])[0]))
            * abs(float(np.diff(dataset.m.values[:2])[0]))
        )
        maximum_beam_fwhm_pixels = max(
            maximum_beam_fwhm_pixels,
            float(beam[0]) / pixel_scale,
        )

    reference = datasets[reference_name]
    reference_t0 = _stokes_i_plane(reference.SKY_RESIDUAL, taylor_term=0)
    y, x = np.indices(reference_t0.shape)
    image_center_y = (reference_t0.shape[0] - 1) / 2.0
    image_center_x = (reference_t0.shape[1] - 1) / 2.0
    central_radius = 0.25 * min(reference_t0.shape)
    central_mask = np.hypot(y - image_center_y, x - image_center_x) <= central_radius
    source_y, source_x = np.unravel_index(
        np.nanargmax(np.where(central_mask, reference_t0, np.nan)),
        reference_t0.shape,
    )

    return {
        "source_y": int(source_y),
        "source_x": int(source_x),
        "maximum_beam_fwhm_pixels": maximum_beam_fwhm_pixels,
        "aperture_radius_pixels": float(aperture_beams * maximum_beam_fwhm_pixels),
        "noise_exclusion_radius_pixels": float(noise_beams * maximum_beam_fwhm_pixels),
    }


def evaluate_configuration(path, geometry, pb_noise_limit, observation, goals):
    """Calculate all requested metrics for one persisted configuration."""
    dataset = xr.open_zarr(path / "continuum.img.zarr")
    manifest = _manifest(path)
    residual_t0 = _stokes_i_plane(dataset.SKY_RESIDUAL, taylor_term=0)
    restored_t0 = _stokes_i_plane(dataset.SKY_RESTORED, taylor_term=0)
    primary_beam = _stokes_i_plane(dataset.PRIMARY_BEAM)
    pbcor_t0 = _stokes_i_plane(dataset.SKY_RESTORED_PBCOR)
    beam = _stokes_i_plane(dataset.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION)

    y, x = np.indices(residual_t0.shape)
    source_distance = np.hypot(
        y - geometry["source_y"],
        x - geometry["source_x"],
    )
    aperture_mask = source_distance <= geometry["aperture_radius_pixels"]
    noise_mask = (
        (primary_beam >= pb_noise_limit)
        & (source_distance >= geometry["noise_exclusion_radius_pixels"])
        & np.isfinite(residual_t0)
    )
    noise_values = residual_t0[noise_mask]
    noise_mad = _robust_sigma(noise_values)
    noise_std = float(np.std(noise_values)) if noise_values.size else np.nan

    delta_l = abs(float(np.diff(dataset.l.values[:2])[0]))
    delta_m = abs(float(np.diff(dataset.m.values[:2])[0]))
    pixel_area_sr = delta_l * delta_m
    beam_area_sr = GAUSSIAN_BEAM_FACTOR * float(beam[0]) * float(beam[1])
    pixels_per_beam = beam_area_sr / pixel_area_sr

    finite_aperture = aperture_mask & np.isfinite(pbcor_t0)
    integrated_flux = float(np.sum(pbcor_t0[finite_aperture]) / pixels_per_beam)
    peak_flux = float(np.max(pbcor_t0[finite_aperture]))
    restored_peak = _maximum_absolute(restored_t0[aperture_mask])
    peak_to_integrated = (
        abs(peak_flux / integrated_flux) if integrated_flux != 0.0 else np.nan
    )
    source_snr_mad = abs(peak_flux) / noise_mad if noise_mad else np.nan
    dynamic_range_std = restored_peak / noise_std if noise_std else np.nan
    off_source_peak = _maximum_absolute(restored_t0[noise_mask])
    off_source_peak_snr = off_source_peak / noise_mad if noise_mad else np.nan

    requested_bandwidth = goals["requested_bandwidth_hz"]
    noise_bandwidth_scaled = noise_std
    if requested_bandwidth is not None:
        if requested_bandwidth <= 0.0:
            raise ValueError("Requested bandwidth must be positive.")
        noise_bandwidth_scaled *= math.sqrt(
            observation["aggregate_bandwidth_hz"] / requested_bandwidth
        )
    noise_qa2_status = _noise_status(
        noise_bandwidth_scaled,
        goals["noise_goal_jy_per_beam"],
        goals["noise_tolerance_fraction"],
    )
    beam_geometric_mean_arcsec = float(np.sqrt(beam[0] * beam[1]) * RAD_TO_ARCSEC)
    resolution_qa2_status = _range_status(
        beam_geometric_mean_arcsec,
        goals["resolution_min_arcsec"],
        goals["resolution_max_arcsec"],
    )
    beam_area_qa2_status = _range_status(
        beam_area_sr * RAD_TO_ARCSEC**2,
        goals["beam_area_min_arcsec2"],
        goals["beam_area_max_arcsec2"],
    )
    flux_qa2_status = _flux_status(
        integrated_flux,
        goals["reference_flux_jy"],
        goals["flux_tolerance_fraction"],
    )
    image_quality_status = (
        "PASS"
        if np.isfinite(peak_to_integrated) and peak_to_integrated >= 0.5
        else "REVIEW"
    )

    if dataset.sizes.get("taylor_term", 1) < 2:
        restored_t1 = np.asarray([], dtype=float)
        spectral_source = "NOT_AVAILABLE_NTERMS_1"
        integrated_spectral_index = np.nan
        peak_spectral_index = np.nan
    else:
        restored_t1 = _stokes_i_plane(dataset.SKY_RESTORED, taylor_term=1)
        if np.any(np.isfinite(restored_t1)):
            spectral_source = "SKY_RESTORED"
            spectral_t0 = _stokes_i_plane(dataset.SKY_RESTORED, taylor_term=0)
            spectral_t1 = restored_t1
        else:
            spectral_source = "SKY_RESIDUAL"
            spectral_t0 = residual_t0
            spectral_t1 = _stokes_i_plane(dataset.SKY_RESIDUAL, taylor_term=1)

        spectral_mask = (
            aperture_mask & np.isfinite(spectral_t0) & np.isfinite(spectral_t1)
        )
        taylor_zero_sum = float(np.sum(spectral_t0[spectral_mask]))
        taylor_one_sum = float(np.sum(spectral_t1[spectral_mask]))
        integrated_spectral_index = (
            taylor_one_sum / taylor_zero_sum if taylor_zero_sum != 0.0 else np.nan
        )
        source_pixel_t0 = float(
            spectral_t0[geometry["source_y"], geometry["source_x"]]
        )
        source_pixel_t1 = float(
            spectral_t1[geometry["source_y"], geometry["source_x"]]
        )
        peak_spectral_index = (
            source_pixel_t1 / source_pixel_t0 if source_pixel_t0 != 0.0 else np.nan
        )
    model = np.asarray(dataset.SKY_MODEL.values)
    experiment = manifest["experiment"]
    use_casa_weights = experiment.get("use_casa_weights")
    weight_implementation = experiment.get("weight_implementation")
    if not weight_implementation:
        weight_implementation = "casa" if use_casa_weights else "unknown"
    return {
        "configuration": path.name,
        "experiment_name": experiment.get("name", path.name),
        "specmode": experiment.get("specmode", "unknown"),
        "weighting_scope": experiment.get("weighting_scope", "unknown"),
        "weight_implementation": weight_implementation,
        "use_casa_weights": use_casa_weights,
        "weight_memory_mode": manifest.get("weight_memory_mode", "in_memory"),
        "widebandpb_memory_mode": manifest.get("widebandpb_memory_mode", "in_memory"),
        "elapsed_seconds": float(manifest["elapsed_seconds"]),
        "beam_major_arcsec": float(beam[0] * RAD_TO_ARCSEC),
        "beam_minor_arcsec": float(beam[1] * RAD_TO_ARCSEC),
        "beam_geometric_mean_arcsec": beam_geometric_mean_arcsec,
        "beam_position_angle_rad": float(beam[2]),
        "beam_area_arcsec2": float(beam_area_sr * RAD_TO_ARCSEC**2),
        "pixels_per_beam": float(pixels_per_beam),
        "noise_n_pixels": int(np.count_nonzero(noise_mask)),
        "residual_noise_mad": noise_mad,
        "residual_noise_std": noise_std,
        "noise_bandwidth_scaled_std": float(noise_bandwidth_scaled),
        "noise_goal_jy_per_beam": goals["noise_goal_jy_per_beam"],
        "noise_tolerance_fraction": goals["noise_tolerance_fraction"],
        "noise_qa2_status": noise_qa2_status,
        "peak_flux_assuming_jy_per_beam": peak_flux,
        "integrated_flux_assuming_jy": integrated_flux,
        "peak_to_integrated_flux_ratio": float(peak_to_integrated),
        "image_quality_status": image_quality_status,
        "source_snr_mad": float(source_snr_mad),
        "dynamic_range_std": float(dynamic_range_std),
        "off_source_peak_snr": float(off_source_peak_snr),
        "resolution_qa2_status": resolution_qa2_status,
        "beam_area_qa2_status": beam_area_qa2_status,
        "flux_qa2_status": flux_qa2_status,
        "reference_flux_jy": goals["reference_flux_jy"],
        "flux_tolerance_fraction": goals["flux_tolerance_fraction"],
        "spectral_index_integrated": float(integrated_spectral_index),
        "spectral_index_source_pixel": float(peak_spectral_index),
        "spectral_index_source_variable": spectral_source,
        "model_nonzero_pixels": int(np.count_nonzero(model)),
        "restored_taylor1_finite_pixels": int(
            np.count_nonzero(np.isfinite(restored_t1))
        ),
        "image_brightness_unit": dataset.SKY_RESTORED.attrs.get("unit", "MISSING"),
    }


def _add_reference_ratios(rows, reference_name):
    """Add dimensionless comparisons against one named configuration."""
    reference = next(row for row in rows if row["configuration"] == reference_name)
    ratio_fields = {
        "beam_area_ratio": "beam_area_arcsec2",
        "noise_ratio": "residual_noise_mad",
        "integrated_flux_ratio": "integrated_flux_assuming_jy",
    }
    for row in rows:
        for output_name, input_name in ratio_fields.items():
            denominator = reference[input_name]
            row[output_name] = row[input_name] / denominator if denominator else np.nan
        row["spectral_index_difference"] = (
            row["spectral_index_integrated"] - reference["spectral_index_integrated"]
        )


def _write_csv(rows, path):
    """Write flat metrics for downstream statistical analysis."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(rows, path):
    """Plot core science and QA2 image-screening metrics."""
    labels = [row["configuration"].replace("_weights", "") for row in rows]
    metrics = (
        ("beam_area_arcsec2", "Beam area (arcsec²)"),
        ("residual_noise_std", "Residual RMS (assumed Jy/beam)"),
        ("dynamic_range_std", "Dynamic range (peak / RMS)"),
        ("integrated_flux_assuming_jy", "Integrated flux (assumed Jy)"),
        ("peak_to_integrated_flux_ratio", "Peak / integrated flux"),
        ("spectral_index_integrated", "Integrated spectral index"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(15, 14), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, metrics, strict=True):
        axis.bar(np.arange(len(rows)), [row[field] for row in rows])
        axis.set_ylabel(label)
        axis.set_xticks(np.arange(len(rows)), labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _preferred_row(rows, **criteria):
    """Choose one deterministic result matching semantic experiment fields."""
    matches = [
        row
        for row in rows
        if all(row.get(field) == value for field, value in criteria.items())
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda row: (
            row["configuration"] != row["experiment_name"],
            "__" in row["configuration"],
            row["configuration"],
        ),
    )[0]


def _comparison_findings(rows):
    """Describe only comparisons supported by the available configurations."""
    findings = []

    for specmode in ("mfs", "mvc"):
        for implementation in ("casa", "proper"):
            local = _preferred_row(
                rows,
                specmode=specmode,
                weighting_scope="local",
                weight_implementation=implementation,
            )
            global_ = _preferred_row(
                rows,
                specmode=specmode,
                weighting_scope="global",
                weight_implementation=implementation,
            )
            if local is not None and global_ is not None:
                change = 100.0 * (
                    local["beam_area_arcsec2"] / global_["beam_area_arcsec2"] - 1.0
                )
                findings.append(
                    f"- Local rather than global {implementation} weighting changes "
                    f"the {specmode.upper()} beam area by `{change:+.2f}%`."
                )

    for scope in ("global", "local"):
        for implementation in ("casa", "proper"):
            mfs = _preferred_row(
                rows,
                specmode="mfs",
                weighting_scope=scope,
                weight_implementation=implementation,
            )
            mvc = _preferred_row(
                rows,
                specmode="mvc",
                weighting_scope=scope,
                weight_implementation=implementation,
            )
            if mfs is not None and mvc is not None:
                noise_change = 100.0 * (
                    mvc["residual_noise_mad"] / mfs["residual_noise_mad"] - 1.0
                )
                flux_change = 100.0 * (
                    mvc["integrated_flux_assuming_jy"]
                    / mfs["integrated_flux_assuming_jy"]
                    - 1.0
                )
                findings.append(
                    f"- MVC relative to MFS changes residual MAD noise by "
                    f"`{noise_change:+.2f}%` and aperture flux by "
                    f"`{flux_change:+.2f}%` for {scope} {implementation} weighting."
                )

    weight_choice_changes = {
        "beam area": [],
        "noise": [],
        "integrated flux": [],
        "spectral index": [],
    }
    for specmode in ("mfs", "mvc"):
        for scope in ("local", "global"):
            casa = _preferred_row(
                rows,
                specmode=specmode,
                weighting_scope=scope,
                weight_implementation="casa",
            )
            proper = _preferred_row(
                rows,
                specmode=specmode,
                weighting_scope=scope,
                weight_implementation="proper",
            )
            if casa is None or proper is None:
                continue
            weight_choice_changes["beam area"].append(
                abs(
                    100.0
                    * (proper["beam_area_arcsec2"] / casa["beam_area_arcsec2"] - 1.0)
                )
            )
            weight_choice_changes["noise"].append(
                abs(
                    100.0
                    * (proper["residual_noise_mad"] / casa["residual_noise_mad"] - 1.0)
                )
            )
            weight_choice_changes["integrated flux"].append(
                abs(
                    100.0
                    * (
                        proper["integrated_flux_assuming_jy"]
                        / casa["integrated_flux_assuming_jy"]
                        - 1.0
                    )
                )
            )
            weight_choice_changes["spectral index"].append(
                abs(
                    proper["spectral_index_integrated"]
                    - casa["spectral_index_integrated"]
                )
            )
    if weight_choice_changes["beam area"]:
        findings.append(
            "- Across available matched CASA/proper weight pairs, the maximum "
            f"changes are `{max(weight_choice_changes['beam area']):.4f}%` in "
            f"beam area, `{max(weight_choice_changes['noise']):.4f}%` in noise, "
            f"`{max(weight_choice_changes['integrated flux']):.4f}%` in integrated "
            "flux, and "
            f"`{max(weight_choice_changes['spectral index']):.3g}` in integrated "
            "spectral index."
        )

    if not findings:
        findings.append(
            "- No matched execution-mode, weighting-scope, or weight-implementation "
            "pairs are available in this result set."
        )
    return findings


def _write_summary(rows, geometry, observation, goals, arguments, path):
    """Write a compact human-readable report with methodological caveats."""
    alpha_values = [row["spectral_index_integrated"] for row in rows]
    model_counts = [row["model_nonzero_pixels"] for row in rows]
    representative_frequency = goals["representative_frequency_hz"]
    if representative_frequency is None:
        representative_product_status = "NOT_ASSESSABLE"
    else:
        representative_product_status = (
            "PASS"
            if any(
                item["minimum_hz"] <= representative_frequency <= item["maximum_hz"]
                for item in observation["frequency_ranges"]
            )
            else "REVIEW"
        )

    def value_or_missing(value, suffix=""):
        return "not supplied" if value is None else f"{value:g}{suffix}"

    lines = [
        "# Continuum demonstrator science metrics",
        "",
        f"Project: `{arguments.project}`  ",
        f"Source variant: `{arguments.variant}`  ",
        f"Reference configuration: `{arguments.reference}`",
        "",
        "## Measurement geometry",
        "",
        f"- Source pixel: `(y={geometry['source_y']}, x={geometry['source_x']})`.",
        f"- Aperture radius: `{geometry['aperture_radius_pixels']:.3f}` pixels.",
        "- Noise region: outside "
        f"`{geometry['noise_exclusion_radius_pixels']:.3f}` pixels and "
        f"within primary beam `{arguments.pb_noise_limit}`.",
        "- Beam area uses the fitted Gaussian `major × minor × π/(4 ln 2)`.",
        "- Integrated flux sums the PB-corrected Taylor-zero image and divides "
        "by pixels per fitted beam.",
        "",
        "## QA2 product and visibility context",
        "",
        f"- Receiver band: `Band {goals['alma_band']}` (selected from "
        f"`{goals['alma_band_source']}`; aggregate reference frequency "
        f"`{observation['reference_frequency_hz'] / 1e9:.6f} GHz`).",
        "- Selected frequency range: "
        f"`{observation['frequency_min_hz'] / 1e9:.6f}` to "
        f"`{observation['frequency_max_hz'] / 1e9:.6f} GHz` across "
        f"`{observation['n_spectral_windows']}` SPWs and "
        f"`{observation['n_channels']}` channels.",
        "- Aggregate selected bandwidth: "
        f"`{observation['aggregate_bandwidth_hz'] / 1e9:.6f} GHz`; channel width "
        f"range: `{observation['channel_width_min_hz'] / 1e6:.6f}` to "
        f"`{observation['channel_width_max_hz'] / 1e6:.6f} MHz`.",
        f"- Representative-product selection: `{representative_product_status}`; "
        "a PI representative frequency was "
        f"`{value_or_missing(representative_frequency, ' Hz')}`.",
        f"- Flag fraction: `{observation['flag_fraction']:.3%}` over "
        f"`{observation['visibility_samples']}` visibility samples.",
        "- On-source duration: "
        f"`{observation['on_source_time_seconds']:.3f} s`; duration retaining any "
        f"unflagged data: `{observation['unflagged_on_source_time_seconds']:.3f} s`.",
        "- Projected uv-radius percentiles (5/50/95%): "
        f"`{observation['uv_radius_p05_m']:.3f}` / "
        f"`{observation['uv_radius_median_m']:.3f}` / "
        f"`{observation['uv_radius_p95_m']:.3f} m`; the 95th percentile is "
        f"`{observation['uv_radius_p95_klambda']:.3f} kλ`.",
        "- Approximate largest angular scale from `0.6 λ / B05`: "
        f"`{observation['largest_angular_scale_estimate_arcsec']:.3f} arcsec`.",
        f"- Input data column: `{observation['source_datacolumn']}`; execution "
        f"blocks present: `{observation['n_execution_blocks']}`.",
        "",
        "## Supplied QA2 goals",
        "",
        "- Noise goal: `"
        f"{value_or_missing(goals['noise_goal_jy_per_beam'], ' Jy/beam')}`; "
        "requested bandwidth: `"
        f"{value_or_missing(goals['requested_bandwidth_hz'], ' Hz')}`.",
        "- Requested geometric-mean resolution range: `"
        f"{value_or_missing(goals['resolution_min_arcsec'], ' arcsec')}` to `"
        f"{value_or_missing(goals['resolution_max_arcsec'], ' arcsec')}`.",
        "- Requested beam-area range: `"
        f"{value_or_missing(goals['beam_area_min_arcsec2'], ' arcsec²')}` to `"
        f"{value_or_missing(goals['beam_area_max_arcsec2'], ' arcsec²')}`.",
        "- External reference flux: `"
        f"{value_or_missing(goals['reference_flux_jy'], ' Jy')}`.",
        "- Applicable guideline allowances: noise `"
        f"{100 * goals['noise_tolerance_fraction']:.0f}%`; flux calibration `"
        f"{100 * goals['flux_tolerance_fraction']:.0f}%`.",
        "",
        "## QA2 image assessment",
        "",
        "PASS/FAIL is assigned only when its required PI goal or external flux "
        "reference was supplied. The peak/integrated diagnostic uses REVIEW, not "
        "FAIL, because resolved targets can legitimately fall below 0.5.",
        "",
        "| Configuration | RMS | Noise | Beam geom. mean | Resolution | Beam area | Area | Source SNR | Dynamic range | Off-src SNR | Peak/int. | Image quality | Flux accuracy |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['configuration']} | {row['noise_bandwidth_scaled_std']:.6g} | "
            f"{row['noise_qa2_status']} | {row['beam_geometric_mean_arcsec']:.6g} | "
            f"{row['resolution_qa2_status']} | {row['beam_area_arcsec2']:.6g} | "
            f"{row['beam_area_qa2_status']} | {row['source_snr_mad']:.6g} | "
            f"{row['dynamic_range_std']:.6g} | {row['off_source_peak_snr']:.6g} | "
            f"{row['peak_to_integrated_flux_ratio']:.6g} | "
            f"{row['image_quality_status']} | {row['flux_qa2_status']} |"
        )
    lines.extend(
        [
            "",
            "## Supplementary science results",
            "",
            "| Configuration | Beam area (arcsec²) | Noise MAD | Integrated flux* | Spectral index† |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['configuration']} | {row['beam_area_arcsec2']:.6g} | "
            f"{row['residual_noise_mad']:.6g} | "
            f"{row['integrated_flux_assuming_jy']:.6g} | "
            f"{row['spectral_index_integrated']:.6g} |"
        )
    spectral_sources = sorted({row["spectral_index_source_variable"] for row in rows})
    lines.extend(
        [
            "",
            "*Flux note:* Flux assumes numerical image values are Jy/beam when "
            "brightness-unit metadata is absent.",
            "",
            "† Spectral-index source products used by this result set: `"
            + "`, `".join(spectral_sources)
            + "`.",
            "",
            "## Available comparisons",
            "",
            "Comparisons below are generated only for semantic experiment pairs "
            "present in this result set; renamed directories are matched through "
            "their completion manifests.",
            "",
        ]
    )
    lines.extend(_comparison_findings(rows))
    lines.extend(
        [
            "- Integrated spectral-index estimates span "
            f"`{min(alpha_values):.5f}` to `{max(alpha_values):.5f}`.",
            "",
            "## Current limitations",
            "",
            "- Model nonzero-pixel counts range from "
            f"`{min(model_counts)}` to `{max(model_counts)}` per configuration. "
            "Zero counts identify dirty-image or otherwise non-cleaning runs.",
            "- Missing brightness-unit metadata limits absolute flux interpretation; "
            "reported fluxes then assume Jy/beam.",
            "- A non-finite restored Taylor term 1 causes that configuration to use "
            "residual Taylor terms for its spectral-index estimate.",
            "- Elapsed times are not controlled performance comparisons unless all "
            "configurations used identical worker and memory settings.",
            "- Calibration flagging quality, bandpass shape, phase-calibrator SNR, "
            "phase transfer, and differential-gain stability require calibration "
            "tables or weblog metrics and are not assessable from these images.",
            "- Renormalization state is not recorded, and off-FOV aliasing is "
            "screened only indirectly through the reported off-source peak SNR.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _evaluate_variant(arguments, project, variant):
    """Evaluate one completed project/source variant and return its output path."""
    variant_arguments = argparse.Namespace(**vars(arguments))
    variant_arguments.project = project
    variant_arguments.variant = variant

    configuration_paths = _configuration_paths(project, variant)
    reference_path = _reference_path(configuration_paths, arguments.reference)
    variant_arguments.reference = reference_path.name
    observation = _observation_metadata(configuration_paths)
    goals = _qa2_goals(variant_arguments, observation)
    if goals["alma_band"] is None:
        raise ValueError(
            "The representative frequency does not identify one unambiguous "
            "ALMA band; supply --alma-band."
        )
    geometry = _geometry(
        configuration_paths,
        reference_path.name,
        arguments.aperture_beams,
        arguments.noise_exclusion_beams,
    )
    rows = [
        evaluate_configuration(
            path,
            geometry,
            arguments.pb_noise_limit,
            observation,
            goals,
        )
        for path in configuration_paths
    ]
    _add_reference_ratios(rows, reference_path.name)

    output_dir = OUTPUT_ROOT / project / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "science_metrics.csv")
    (output_dir / "science_metrics.json").write_text(
        json.dumps(
            {
                "guideline": "ALMA QA2 Cycle 13, section 11.5",
                "reference_configuration": reference_path.name,
                "geometry": geometry,
                "observation": observation,
                "qa2_goals": goals,
                "metrics": rows,
                "not_assessed_from_current_products": [
                    "calibration flagging character and raw/calibrated outliers",
                    "bandpass flatness, SNR, and line contamination",
                    "phase-calibrator SNR and phase-transfer stability",
                    "differential-gain calibration stability",
                    "renormalization correction",
                    "direct bright-source/aliasing inspection outside the FOV",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    _write_plot(rows, output_dir / "science_metrics.png")
    _write_summary(
        rows,
        geometry,
        observation,
        goals,
        variant_arguments,
        output_dir / "summary.md",
    )
    print(f"Wrote science evaluation to {output_dir}")
    return output_dir


def main():
    """Evaluate one result variant or every completed variant below results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="evaluate every project/variant with completed AstroVIPER results",
    )
    parser.add_argument("--project", default="2023.1.01342.S")
    parser.add_argument("--variant", default="mfs_0_7_11_15_18")
    parser.add_argument(
        "--reference",
        help=(
            "reference result directory or manifest experiment name; defaults "
            "to the preferred available MFS configuration"
        ),
    )
    parser.add_argument("--aperture-beams", type=float, default=3.0)
    parser.add_argument("--noise-exclusion-beams", type=float, default=5.0)
    parser.add_argument("--pb-noise-limit", type=float, default=0.3)
    parser.add_argument("--alma-band", type=int, choices=range(1, 11))
    parser.add_argument("--noise-goal-jy-per-beam", type=float)
    parser.add_argument("--requested-bandwidth-hz", type=float)
    parser.add_argument("--representative-frequency-hz", type=float)
    parser.add_argument("--resolution-min-arcsec", type=float)
    parser.add_argument("--resolution-max-arcsec", type=float)
    parser.add_argument("--beam-area-min-arcsec2", type=float)
    parser.add_argument("--beam-area-max-arcsec2", type=float)
    parser.add_argument("--reference-flux-jy", type=float)
    args = parser.parse_args()

    if not args.all:
        _evaluate_variant(args, args.project, args.variant)
        return

    variants = _completed_variants()
    if not variants:
        parser.error(f"no completed AstroVIPER results found below {RESULTS_ROOT}")

    successes = []
    failures = []
    for project, variant in variants:
        try:
            output_dir = _evaluate_variant(args, project, variant)
            successes.append(
                {"project": project, "variant": variant, "output": str(output_dir)}
            )
        except (
            Exception
        ) as error:  # continue so one damaged result does not hide others
            failures.append(
                {"project": project, "variant": variant, "error": str(error)}
            )
            print(
                f"Evaluation failed for {project}/{variant}: {error}",
                file=sys.stderr,
            )

    all_summary = {"evaluated": successes, "failed": failures}
    (OUTPUT_ROOT / "all_evaluations.json").write_text(
        json.dumps(all_summary, indent=2) + "\n"
    )
    print(
        f"Evaluation sweep complete: {len(successes)} succeeded, "
        f"{len(failures)} failed."
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
