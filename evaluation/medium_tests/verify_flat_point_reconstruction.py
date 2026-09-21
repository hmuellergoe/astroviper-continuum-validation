"""Evaluate the phase-centre single-term example without relaxing failed checks."""

from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def evaluate(root):
    rows = []
    for stage in ["dirty", "update", "clean"]:
        files = {
            engine: np.load(root / f"{engine}_{stage}.npz")
            for engine in ["casa", "astroviper"]
        }
        for key in ["model", "residual", "psf", "pb", "restored", "pbcor"]:
            a = files["casa"][key]
            b = files["astroviper"][key]
            if a.shape != (200, 200) or b.shape != a.shape:
                raise ValueError(
                    f"{stage}/{key}: unexpected image shapes {a.shape}, {b.shape}"
                )
            rows.append(
                dict(
                    stage=stage,
                    check="cross_code_" + key,
                    max_abs=float(np.nanmax(abs(a - b))),
                    passed=bool(
                        np.allclose(a, b, rtol=0.05, atol=0.005, equal_nan=True)
                    ),
                )
            )
        for engine, f in files.items():
            model = f["model"]
            res = f["residual"]
            dirty = np.load(root / f"{engine}_dirty.npz")["residual"]
            # This fixture has an exact one-pixel model: its predicted dirty
            # response is model_flux times the initial unit-source response.
            outside = model.copy()
            outside[100, 100] = 0
            supported = bool(np.max(abs(outside)) < 1e-10)
            predicted = (1.0 - model[100, 100]) * dirty
            rows.append(
                dict(
                    stage=stage,
                    check=engine + "_prediction",
                    max_abs=float(np.max(abs(res - predicted))),
                    passed=supported
                    and bool(np.allclose(res, predicted, rtol=0.05, atol=0.005)),
                )
            )
            if stage == "clean":
                peak = float(np.max(abs(res)))
                rows.append(
                    dict(
                        stage=stage,
                        check=engine + "_threshold",
                        value=peak,
                        passed=peak <= 0.01,
                    )
                )
                rows.append(
                    dict(
                        stage=stage,
                        check=engine + "_unit_peak",
                        value=float(f["restored"][100, 100]),
                        passed=abs(float(f["restored"][100, 100]) - 1.0) <= 0.01,
                    )
                )
            if stage == "dirty":
                rows.append(
                    dict(
                        stage=stage,
                        check=engine + "_unit_dirty",
                        value=float(res[100, 100]),
                        passed=abs(float(res[100, 100]) - 1.0) <= 0.001,
                    )
                )
    (root / "checks.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(r)
    return all(r["passed"] for r in rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results", type=Path, default=ROOT / "results/medium_tests/flat_point"
    )
    a = p.parse_args()
    raise SystemExit(0 if evaluate(a.results) else 1)
