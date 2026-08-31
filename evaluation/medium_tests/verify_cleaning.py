"""Verify that the shared CASA/AstroVIPER threshold causes real cleaning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "scripts" / "medium_tests"))
from cases import CLEANING_CASE, weighting_suffix  # noqa: E402


def _finite_abs_peak(values):
    """Return the largest absolute finite value in an array."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise AssertionError("Product contains no finite values.")
    return float(np.max(np.abs(finite)))


def verify_implementation(name, dirty_path, clean_path):
    """Require a nonzero clean model and a lower Taylor-zero residual peak."""
    dirty = np.load(dirty_path)
    clean = np.load(clean_path)
    if "model_t0" not in clean:
        raise AssertionError(f"{name} clean output does not contain model_t0.")

    model = np.asarray(clean["model_t0"])
    finite_model = model[np.isfinite(model)]
    nonzero_model_pixels = int(np.count_nonzero(finite_model))
    model_peak = _finite_abs_peak(model)
    dirty_peak = _finite_abs_peak(dirty["residual_t0"])
    clean_peak = _finite_abs_peak(clean["residual_t0"])

    if nonzero_model_pixels == 0 or model_peak == 0.0:
        raise AssertionError(f"{name} did not put any structure into its model.")
    if clean_peak >= dirty_peak:
        raise AssertionError(
            f"{name} residual peak did not decrease: {dirty_peak} -> {clean_peak}."
        )

    print(
        f"{name}: model pixels={nonzero_model_pixels}, model peak={model_peak:.6g}, "
        f"residual peak={dirty_peak:.6g} -> {clean_peak:.6g}"
    )


def main():
    """Validate one matching dirty/clean CASA and AstroVIPER result pair."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=HARNESS_ROOT / "results" / "medium_tests",
    )
    parser.add_argument("--case", default="phase_center")
    parser.add_argument("--mode", choices=("mfs", "mvc"), default="mfs")
    parser.add_argument("--weighting", default="natural_local")
    args = parser.parse_args()

    suffix = weighting_suffix(args.weighting)
    for implementation, prefix in (("CASA", "casa"), ("AstroVIPER", "radps")):
        base = f"{prefix}_{args.case}_{args.mode}{suffix}"
        verify_implementation(
            implementation,
            args.results / f"{base}.npz",
            args.results / f"{base}_clean.npz",
        )
    print(
        "PASS: both implementations cleaned structure above the shared "
        f"{CLEANING_CASE['threshold_jy']} Jy threshold."
    )


if __name__ == "__main__":
    main()
