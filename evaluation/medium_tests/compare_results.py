"""Compare neutral CASA and RADPS continuum major-cycle products."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS_ROOT / "scripts" / "medium_tests"))
from cases import WEIGHTING_CASES, weighting_suffix  # noqa: E402

PRODUCTS = ("residual_t0", "residual_t1", "psf_t0", "psf_t1", "psf_t2", "pb_t0")


def plane(array):
    """Select scalar time/polarization axes and retain the two image axes."""
    result = np.real_if_close(np.asarray(array)).squeeze()
    # RADPS currently returns both linear-correlation/Stokes planes even when
    # the requested output is scalar I. The first plane is Stokes I.
    while result.ndim > 2:
        result = result[0]
    if result.ndim != 2:
        raise ValueError(
            f"Expected a 2-D scalar image after squeeze; got {result.shape}"
        )
    return np.asarray(result, dtype=np.float64)


def orientations(array):
    """Candidate axis conventions for comparing CASA and xradio images."""
    return {
        "identity": array,
        "transpose": array.T,
        "flip0": array[::-1, :],
        "flip1": array[:, ::-1],
        "flip01": array[::-1, ::-1],
        "transpose_flip0": array.T[::-1, :],
        "transpose_flip1": array.T[:, ::-1],
        "transpose_flip01": array.T[::-1, ::-1],
    }


def metrics(reference, candidate):
    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not finite.any():
        return {"rms": np.inf, "max_abs": np.inf, "relative_rms": np.inf}
    delta = candidate[finite] - reference[finite]
    rms = float(np.sqrt(np.mean(delta**2)))
    scale = float(np.sqrt(np.mean(reference[finite] ** 2)))
    return {
        "rms": rms,
        "max_abs": float(np.max(np.abs(delta))),
        "relative_rms": rms / max(scale, np.finfo(float).eps),
    }


def compare_pair(casa_path, radps_path, rtol, atol, enforce_tolerance=True):
    casa = np.load(casa_path)
    radps = np.load(radps_path)
    failures = []
    chosen_orientation = None
    for name in PRODUCTS:
        reference = plane(casa[name])
        raw_candidate = plane(radps[name])
        candidates = orientations(raw_candidate)
        if chosen_orientation is None:
            scored = {
                key: metrics(reference, value) for key, value in candidates.items()
            }
            chosen_orientation = min(scored, key=lambda key: scored[key]["rms"])
        candidate = candidates[chosen_orientation]
        result = metrics(reference, candidate)
        peak_ref = np.unravel_index(np.nanargmax(np.abs(reference)), reference.shape)
        peak_test = np.unravel_index(np.nanargmax(np.abs(candidate)), candidate.shape)
        passed = np.allclose(candidate, reference, rtol=rtol, atol=atol, equal_nan=True)
        status = "PASS" if passed else "FAIL"
        if not enforce_tolerance and not passed:
            status = "EXPECTED-DIFFERENCE"
        print(
            f"{name:12s} orientation={chosen_orientation:18s} "
            f"rms={result['rms']:.6g} rel_rms={result['relative_rms']:.6g} "
            f"max={result['max_abs']:.6g} peaks={peak_ref}/{peak_test} "
            f"{status}"
        )
        if enforce_tolerance and not passed:
            failures.append(name)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=HARNESS_ROOT / "results" / "medium_tests",
    )
    parser.add_argument("--cases", nargs="+", default=["phase_center"])
    parser.add_argument(
        "--modes", nargs="+", choices=("mfs", "mvc"), default=["mfs", "mvc"]
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        choices=WEIGHTING_CASES,
        default=list(WEIGHTING_CASES),
        help=(
            "Weighting cases to compare. By default, compare natural-local, "
            "Briggs-local, and Briggs-global."
        ),
    )
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()
    all_failures = []
    for case in args.cases:
        for mode in args.modes:
            for weighting in args.weightings:
                suffix = weighting_suffix(weighting)
                print(f"\n=== {case}/{mode}/{weighting}: CASA versus RADPS ===")
                failures = compare_pair(
                    args.results / f"casa_{case}_{mode}{suffix}.npz",
                    args.results / f"radps_{case}_{mode}{suffix}.npz",
                    args.rtol,
                    args.atol,
                    enforce_tolerance=(weighting != "briggs_local"),
                )
                all_failures.extend(
                    f"{case}/{mode}/{weighting}/{name}" for name in failures
                )
    if all_failures:
        raise SystemExit("Products outside tolerance: " + ", ".join(all_failures))


if __name__ == "__main__":
    main()
