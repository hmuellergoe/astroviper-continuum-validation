"""Unit tests for ALMA QA2 criteria used by the continuum evaluator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

import evaluate_science_metrics as evaluator
from evaluate_science_metrics import (
    _channel_width_hz,
    _comparison_findings,
    _completed_variants,
    _flux_status,
    _infer_alma_band,
    _maximum_absolute,
    _noise_status,
    _range_status,
    _reference_path,
)


class TestQa2Criteria(unittest.TestCase):
    """Check band inference and PASS/FAIL/NOT_ASSESSABLE boundaries."""

    def test_band_and_channel_width_metadata(self):
        """Band 1 and XRADIO scalar channel-width metadata are recognized."""
        dataset = xr.Dataset(coords={"frequency": [38.0e9, 38.1e9]})
        dataset.frequency.attrs["channel_width"] = {"data": 15.625e6}
        self.assertEqual(_infer_alma_band(40.0e9), 1)
        self.assertEqual(_channel_width_hz(dataset), 15.625e6)

    def test_noise_uses_band_allowance_as_upper_limit(self):
        """Noise better than the goal passes and excess beyond allowance fails."""
        self.assertEqual(_noise_status(1.09, 1.0, 0.10), "PASS")
        self.assertEqual(_noise_status(1.11, 1.0, 0.10), "FAIL")
        self.assertEqual(_noise_status(1.0, None, 0.10), "NOT_ASSESSABLE")

    def test_resolution_and_flux_boundaries(self):
        """Inclusive beam ranges and symmetric flux tolerances are enforced."""
        self.assertEqual(_range_status(2.0, 2.0, 3.0), "PASS")
        self.assertEqual(_range_status(3.1, 2.0, 3.0), "FAIL")
        self.assertEqual(_range_status(2.0, None, None), "NOT_ASSESSABLE")
        self.assertEqual(_flux_status(0.95, 1.0, 0.05), "PASS")
        self.assertEqual(_flux_status(1.051, 1.0, 0.05), "FAIL")
        self.assertEqual(_flux_status(1.0, None, 0.05), "NOT_ASSESSABLE")

    def test_empty_image_selection_returns_nan(self):
        """An empty QA2 noise region is reported as unavailable, not an error."""
        self.assertTrue(np.isnan(_maximum_absolute([])))


class TestResultDiscovery(unittest.TestCase):
    """Check renamed configuration discovery and all-result orchestration."""

    @staticmethod
    def _write_manifest(path, name, specmode="mfs", scope="global", casa=False):
        path.mkdir(parents=True)
        (path / "complete.json").write_text(
            json.dumps(
                {
                    "experiment": {
                        "name": name,
                        "specmode": specmode,
                        "weighting_scope": scope,
                        "weight_implementation": "casa" if casa else "proper",
                        "use_casa_weights": casa,
                    }
                }
            )
        )

    def test_completed_variants_and_renamed_reference_use_manifests(self):
        """Result suffixes do not hide variants or canonical experiment names."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_root = Path(temporary_directory)
            result = (
                results_root
                / "project"
                / "variant"
                / "astroviper"
                / "mfs_global_proper_weights__weights_in_place"
            )
            self._write_manifest(result, "mfs_global_proper_weights")
            with patch.object(evaluator, "RESULTS_ROOT", results_root):
                self.assertEqual(_completed_variants(), [("project", "variant")])
            self.assertEqual(
                _reference_path([result], "mfs_global_proper_weights"), result
            )
            self.assertEqual(_reference_path([result]), result)

    def test_partial_results_generate_only_available_comparisons(self):
        """A report with only a matched local/global pair remains evaluable."""
        common = {
            "experiment_name": "renamed",
            "specmode": "mfs",
            "weight_implementation": "proper",
            "beam_area_arcsec2": 2.0,
            "residual_noise_mad": 1.0,
            "integrated_flux_assuming_jy": 1.0,
            "spectral_index_integrated": 0.0,
        }
        rows = [
            {**common, "configuration": "global_new_name", "weighting_scope": "global"},
            {
                **common,
                "configuration": "local_new_name",
                "weighting_scope": "local",
                "beam_area_arcsec2": 2.2,
            },
        ]
        findings = _comparison_findings(rows)
        self.assertTrue(any("MFS beam area" in finding for finding in findings))

    def test_all_option_evaluates_each_discovered_variant(self):
        """The --all CLI dispatches every discovered project/variant pair."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            variants = [("project_a", "variant_a"), ("project_b", "variant_b")]
            with (
                patch.object(evaluator, "OUTPUT_ROOT", output_root),
                patch.object(evaluator, "_completed_variants", return_value=variants),
                patch.object(
                    evaluator,
                    "_evaluate_variant",
                    side_effect=[output_root / "a", output_root / "b"],
                ) as evaluate,
                patch.object(sys, "argv", ["evaluate_science_metrics.py", "--all"]),
            ):
                evaluator.main()
            self.assertEqual(evaluate.call_count, 2)
            summary = json.loads((output_root / "all_evaluations.json").read_text())
            self.assertEqual(len(summary["evaluated"]), 2)
            self.assertEqual(summary["failed"], [])


if __name__ == "__main__":
    unittest.main()
