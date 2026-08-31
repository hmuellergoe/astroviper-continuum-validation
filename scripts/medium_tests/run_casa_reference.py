"""Generate CASA major-cycle reference products for MFS and MVC.

Run with a CASA Python environment. The default ``niter=0`` mode isolates dirty
residual/PSF/PB construction; ``--clean`` exercises a thresholded minor loop.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from casatasks import tclean
from casatools import image

SCRIPT_ROOT = Path(__file__).resolve().parent
HARNESS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
from cases import (  # noqa: E402
    CASES,
    CLEANING_CASE,
    WEIGHTING_CASES,
    cleaning_suffix,
    selected_cases,
    selected_weightings,
    weighting_suffix,
)


def read_casa_image(path: Path) -> np.ndarray:
    """Read a CASA image and return its raw numpy array."""
    tool = image()
    tool.open(str(path))
    try:
        return np.asarray(tool.getchunk())
    finally:
        tool.close()


def remove_products(prefix: Path) -> None:
    for path in prefix.parent.glob(prefix.name + "*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def export_products(prefix: Path, destination: Path) -> None:
    names = {
        "residual_t0": prefix.with_name(prefix.name + ".residual.tt0"),
        "residual_t1": prefix.with_name(prefix.name + ".residual.tt1"),
        "psf_t0": prefix.with_name(prefix.name + ".psf.tt0"),
        "psf_t1": prefix.with_name(prefix.name + ".psf.tt1"),
        "psf_t2": prefix.with_name(prefix.name + ".psf.tt2"),
        "pb_t0": prefix.with_name(prefix.name + ".pb.tt0"),
    }
    for term in (0, 1):
        model = prefix.with_name(prefix.name + f".model.tt{term}")
        if model.exists():
            names[f"model_t{term}"] = model
    missing = [str(path) for path in names.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"CASA did not create expected products: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{name: read_casa_image(path) for name, path in names.items()},
    )


def run_case(
    case_name: str,
    config: dict,
    mode: str,
    weighting_name: str,
    weighting_config: dict,
    data_dir: Path,
    out_dir: Path,
    clean: bool,
):
    ms_path = data_dir / config["ms"]
    suffix = weighting_suffix(weighting_name)
    clean_suffix = cleaning_suffix(clean)
    prefix = out_dir / "casa_images" / f"{case_name}_{mode}{suffix}{clean_suffix}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    remove_products(prefix)

    kwargs = dict(
        vis=str(ms_path),
        imagename=str(prefix),
        field=config["field"],
        imsize=config["imsize"],
        cell=f'{config["cell_arcsec"]}arcsec',
        reffreq=f'{config["reffreq_hz"]}Hz',
        specmode=mode,
        gridder="standard",
        deconvolver="mtmfs",
        nterms=2,
        weighting=weighting_config["casa_weighting"],
        niter=CLEANING_CASE["niter"] if clean else 0,
        threshold=(f'{CLEANING_CASE["threshold_jy"]}Jy' if clean else "0Jy"),
        calcpsf=True,
        calcres=True,
        restoration=False,
        pblimit=config["pblimit"],
        parallel=False,
    )
    if config["phasecenter"] is not None:
        kwargs["phasecenter"] = config["phasecenter"]
    if mode == "mvc":
        kwargs["nchan"] = config["nchan_mvc"]
    if weighting_config["robust"] is not None:
        kwargs["robust"] = weighting_config["robust"]

    print(f"Running CASA {case_name}/{mode}/{weighting_name}: {config['description']}")
    tclean(**kwargs)
    export_products(
        prefix,
        out_dir / f"casa_{case_name}_{mode}{suffix}{clean_suffix}.npz",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=HARNESS_ROOT / "data" / "medium_tests"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HARNESS_ROOT / "results" / "medium_tests",
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=["phase_center"])
    parser.add_argument(
        "--modes", nargs="+", choices=("mfs", "mvc"), default=["mfs", "mvc"]
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run the shared 0.1 Jy threshold-cleaning regression.",
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        choices=WEIGHTING_CASES,
        default=["natural_local"],
    )
    args = parser.parse_args()
    for case_name, config in selected_cases(args.cases):
        for mode in args.modes:
            for weighting_name, weighting_config in selected_weightings(
                args.weightings
            ):
                run_case(
                    case_name,
                    config,
                    mode,
                    weighting_name,
                    weighting_config,
                    args.data_dir,
                    args.output_dir,
                    args.clean,
                )


if __name__ == "__main__":
    main()
