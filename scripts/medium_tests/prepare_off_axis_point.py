"""Create a single-field, PB-sensitive off-axis point-source fixture.

Run this script with a CASA Python interpreter.  It copies ``refim_point.ms``
and replaces its visibility data with a one-Jy, flat-spectrum point source
offset north of the shared field, pointing, and image phase center.  Therefore
the regression exercises direction-dependent primary-beam behavior without
requiring either mosaic imaging or a visibility phase-center shift.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from casatasks import delmod
from casatools import componentlist, measures, quanta, simulator, table

HARNESS_ROOT = Path(__file__).resolve().parents[2]


def field_phase_direction(ms_path: Path) -> tuple[float, float]:
    """Return the sole FIELD phase direction as J2000 radians."""
    tool = table()
    try:
        tool.open(str(ms_path / "FIELD"))
        if tool.nrows() != 1:
            raise RuntimeError(
                f"The source fixture must have one FIELD row, found {tool.nrows()}."
            )
        direction = np.asarray(tool.getcol("PHASE_DIR"))[:, 0, 0]
    finally:
        tool.close()
    return float(direction[0]), float(direction[1])


def validate_single_field_ms(ms_path: Path) -> None:
    """Require one field and finite, PB-attenuated visibility data."""
    tool = table()
    amplitude_min = np.inf
    amplitude_max = -np.inf
    all_finite = True
    try:
        tool.open(str(ms_path / "FIELD"))
        field_rows = tool.nrows()
        tool.close()

        tool.open(str(ms_path))
        field_ids = set(int(value) for value in tool.getcol("FIELD_ID"))
        for startrow in range(0, tool.nrows(), 4096):
            nrow = min(4096, tool.nrows() - startrow)
            sample = np.asarray(tool.getcol("DATA", startrow=startrow, nrow=nrow))
            all_finite &= bool(np.isfinite(sample).all())
            amplitudes = np.abs(sample)
            amplitude_min = min(amplitude_min, float(np.nanmin(amplitudes)))
            amplitude_max = max(amplitude_max, float(np.nanmax(amplitudes)))
    finally:
        tool.close()

    if field_rows != 1 or field_ids != {0}:
        raise RuntimeError(
            "The off-axis fixture is not single-field: "
            f"FIELD rows={field_rows}, FIELD_ID values={sorted(field_ids)}."
        )
    if not all_finite or not 0.0 < amplitude_min < amplitude_max < 1.0:
        raise RuntimeError(
            "The synthetic point-source visibilities are invalid: "
            f"amplitude range=({amplitude_min}, {amplitude_max})."
        )


def replace_data_with_model(ms_path: Path, component_path: Path) -> None:
    """Predict PB-attenuated component visibilities into the observed data."""
    tool = simulator()
    try:
        tool.openfromms(str(ms_path))
        tool.setvp(dovp=True, usedefaultvp=True)
        tool.predict(complist=str(component_path), incremental=False)
    finally:
        tool.done()
    # The prediction belongs only in observed DATA/CORRECTED_DATA. Remove the
    # scratch model state so tclean starts from an empty sky model.
    delmod(vis=str(ms_path), otf=True, scr=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=HARNESS_ROOT / "data" / "medium_tests" / "refim_point.ms",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HARNESS_ROOT / "data" / "medium_tests" / "refim_offaxis_point.ms",
    )
    parser.add_argument(
        "--offset-arcsec",
        type=float,
        default=600.0,
        help="Point-source offset north of the pointing/image center.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise FileNotFoundError(f"Input Measurement Set not found: {args.input}")
    if args.output.exists():
        if not args.overwrite:
            validate_single_field_ms(args.output)
            print(f"Single-field off-axis fixture already exists: {args.output}")
            return
        shutil.rmtree(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.input, args.output)

    ra_rad, dec_rad = field_phase_direction(args.output)
    offset_rad = np.deg2rad(args.offset_arcsec / 3600.0)
    direction = measures().direction(
        "J2000",
        quanta().quantity(ra_rad, "rad"),
        quanta().quantity(dec_rad + offset_rad, "rad"),
    )
    component_path = args.output.parent / f".{args.output.name}.component.cl"
    if component_path.exists():
        shutil.rmtree(component_path)
    components = componentlist()
    try:
        components.addcomponent(
            flux=1.0,
            fluxunit="Jy",
            dir=direction,
            shape="point",
            spectrumtype="spectral index",
            index=0.0,
        )
        components.rename(str(component_path))
    finally:
        components.close()

    try:
        replace_data_with_model(args.output, component_path)
    finally:
        if component_path.exists():
            shutil.rmtree(component_path)

    validate_single_field_ms(args.output)
    print(
        f"Created one-Jy point source {args.offset_arcsec:g} arcsec north of the "
        f"single pointing/image center: {args.output}"
    )


if __name__ == "__main__":
    main()
