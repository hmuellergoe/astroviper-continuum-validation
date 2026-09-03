#!/usr/bin/env python3
"""Evaluate every retained CASA result against the corresponding AstroViper run."""

from __future__ import annotations

import json
from pathlib import Path

from compare_casa_astroviper import (
    RESULTS,
    ROOT,
    _load_casa,
    _metrics,
    _render_comparison,
)


GS8_VARIANT = "mfs_0_25_27_29_31_45_47_49_51_65_67_69_71_85_87_89_91"
CASES = (
    ("2017.1.00661.S", "mfs_0_16", "casa_benchmark_astroviper_matched", False),
    ("2017.1.00999.S", "mfs_0_25", "casa_benchmark_astroviper_matched", False),
    ("2018.1.01079.S", GS8_VARIANT, "casa_benchmark_astroviper_matched", False),
    ("2018.1.01079.S", GS8_VARIANT, "casa_benchmark__automask_2sigma", True),
    ("2018.1.01079.S", GS8_VARIANT, "casa_benchmark__manual_final_mask_2sigma", True),
    ("2018.1.01284.S", "mfs_0_16_18_20_22", "casa_benchmark_astroviper_matched", False),
    ("2019.1.00678.S", "mfs_0_7_21_23_25", "casa_benchmark_astroviper_matched", False),
    ("2023.1.01342.S", "mfs_0_7_11_15_18", "casa_benchmark_astroviper_matched", False),
)
OUTPUT_JSON = ROOT / "evaluation" / "casa_algorithm_matched_results.json"
OUTPUT_MD = ROOT / "evaluation" / "casa_algorithm_matched_report.md"


def _paths(directory: Path) -> dict[str, Path]:
    return {
        "restored": directory / "continuum.image.fits",
        "restored_pbcor": directory / "continuum.image.pbcor.fits",
        "residual": directory / "continuum.residual.fits",
        "model": directory / "continuum.model.fits",
        "psf": directory / "continuum.psf.fits",
        "pb": directory / "continuum.pb.fits",
    }


def _matching_av_paths(base: Path, masked: bool) -> list[Path]:
    paths = []
    for pattern in (
        "mfs_global_casa_weights*/continuum.img.zarr",
        "mvc_global_casa_weights*/continuum.img.zarr",
    ):
        for av_path in sorted((base / "astroviper").glob(pattern)):
            manifest = json.loads((av_path.parent / "complete.json").read_text())
            has_mask = manifest.get("clean_mask") is not None
            if has_mask == masked:
                paths.append(av_path)
    return paths


def main() -> None:
    results = []
    for project, variant, benchmark_name, av_masked in CASES:
        base = RESULTS / project / variant
        benchmark = base / benchmark_name
        complete = json.loads((benchmark / "complete.json").read_text())
        configuration = complete["configuration"]
        if configuration["deconvolver"] != "hogbom" or configuration["nterms"] != 1:
            raise RuntimeError(f"Noncanonical retained CASA result: {benchmark}")
        casa_paths = _paths(benchmark)
        casa = _load_casa(casa_paths)
        comparisons = []
        for av_path in _matching_av_paths(base, av_masked):
            metrics, av = _metrics(base, av_path, casa, casa_paths)
            metrics["casa_benchmark"] = benchmark_name
            plot = (
                ROOT
                / "evaluation"
                / project
                / variant
                / f"{benchmark_name}_vs_{av_path.parent.name}.png"
            )
            _render_comparison(
                plot,
                casa,
                av,
                f"{project} / {variant}\n{benchmark_name} vs {av_path.parent.name}",
            )
            metrics["plot"] = str(plot)
            comparisons.append(metrics)
        results.append(
            {
                "project": project,
                "variant": variant,
                "casa_benchmark": benchmark_name,
                "astroviper_masked": av_masked,
                "configuration": configuration,
                "comparisons": comparisons,
            }
        )
    OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    _write_report(results)
    print(OUTPUT_JSON)
    print(OUTPUT_MD)


def _write_report(results: list[dict]) -> None:
    lines = [
        "# Retained CASA–AstroViper comparisons",
        "",
        "This report covers every retained CASA dataset in the result tree. All "
        "retained CASA runs use `deconvolver=hogbom` and `nterms=1`. The standard "
        "six-example comparison is unmasked on both sides. GS8 additionally "
        "retains the 83 microJy auto-multithresh mask-generation run and the final "
        "manual-mask run; those are compared with the fixed-mask AstroViper result.",
        "",
        "| Project | CASA result | AV execution | Pairing | PSF corr. | Model corr. | Restored corr. | Relative L2 | Peak ratio | Residual MAD CASA/AV | Model Jy CASA/AV |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        pairing = "fixed mask" if result["astroviper_masked"] else "unmasked"
        for item in result["comparisons"]:
            lines.append(
                f"| {result['project']} | `{result['casa_benchmark']}` | "
                f"`{item['astroviper_run'].split('__')[0]}` | {pairing} | "
                f"{item['psf_correlation']:.6f} | "
                f"{item['model_correlation']:.6f} | "
                f"{item['restored_correlation']:.5f} | "
                f"{item['restored_relative_l2']:.3f} | "
                f"{item['peak_ratio_astroviper_over_casa']:.3f} | "
                f"{item['casa_residual_mad_sigma_jy_per_beam']:.4g}/"
                f"{item['astroviper_residual_mad_sigma_jy_per_beam']:.4g} | "
                f"{item['casa_model_sum_jy']:.5g}/"
                f"{item['astroviper_model_sum_jy']:.5g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The unmasked rows are the closest direct CASA comparison to the current "
            "AstroViper minor cycle.",
            "- HOPS is restricted on both sides to original-MS FIELD 63, the "
            "central pointing at the requested image phase center; CASA uses the "
            "standard single-field gridder with `mosweight=False`.",
            "- The GS8 final manual mask contains 349 pixels. The retained Hogbom "
            "2-sigma automask also contains 349 pixels, but shares 341 pixels with "
            "the exact final mask, so its comparison is a provenance diagnostic rather "
            "than an exact same-mask acceptance test.",
            "- Primary-beam-corrected edges can differ because CASA and AstroViper use "
            "different primary-beam implementations.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
