"""Create an exact flat-spectrum, one-Jy Stokes-I phase-centre fixture."""

from pathlib import Path
import argparse
import shutil
import json
import numpy as np
from casatools import table
from casatasks import delmod
from prepare_off_axis_point import field_phase_direction

ROOT = Path(__file__).resolve().parents[2]


def prepare(source, output):
    if output.exists():
        raise FileExistsError(f"{output}: choose a new output or remove it explicitly")
    field_phase_direction(source)  # Reject multi-field input.
    t = table()
    t.open(str(source / "POLARIZATION"))
    correlations = t.getcol("CORR_TYPE")
    t.close()
    if not np.isin(correlations, [5, 8, 9, 12]).all():
        raise ValueError("Fixture requires parallel-hand correlations only")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    # At the phase centre, a one-Jy unpolarized point has V=1+0j on
    # every parallel hand and frequency. No noise or spectral slope is added.
    t.open(str(output), nomodify=False)
    try:
        if set(t.getcol("FIELD_ID")) != {0}:
            raise ValueError("Expected field 0 only")
        for start in range(0, t.nrows(), 4096):
            n = min(4096, t.nrows() - start)
            shape = t.getcol("DATA", startrow=start, nrow=n).shape
            data = np.ones(shape, dtype=np.complex128)
            for col in ["DATA", "CORRECTED_DATA"]:
                if col in t.colnames():
                    t.putcol(col, data, startrow=start, nrow=n)
    finally:
        t.close()
    delmod(vis=str(output), otf=True, scr=True)
    t.open(str(output))
    try:
        for start in range(0, t.nrows(), 4096):
            values = t.getcol("DATA", startrow=start, nrow=min(4096, t.nrows() - start))
            np.testing.assert_array_equal(values, np.ones_like(values))
    finally:
        t.close()
    (output.parent / (output.name + ".truth.json")).write_text(
        json.dumps(
            dict(
                flux_jy=1.0,
                spectral_index=0.0,
                offset_arcsec=0.0,
                input=str(source),
                phase_direction_rad=field_phase_direction(output),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=ROOT / "data/medium_tests/refim_point.ms"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "data/medium_tests/refim_flat_point.ms"
    )
    a = p.parse_args()
    prepare(a.input.resolve(), a.output.resolve())
    print("Flat one-Jy fixture verified:", a.output)
