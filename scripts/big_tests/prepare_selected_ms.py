#!/usr/bin/env python3
"""Apply each tclean selection and form one MSv2 for XRADIO conversion."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from continuum_common import (
    DATA_ROOT,
    PROJECTS,
    RESULTS_ROOT,
    as_list,
    project_variants,
    resolve_available_config,
    write_manifest,
)


def _remove_if_requested(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{path} exists; use --overwrite to replace it")
    shutil.rmtree(path)


def prepare_project(project: str, overwrite: bool) -> None:
    """Create selected, concatenated MS inputs matching each MFS configuration."""
    from casatasks import concat, mstransform

    data_dir = DATA_ROOT / project
    for variant, config in project_variants(project):
        prep_dir = RESULTS_ROOT / project / variant / "prepared"
        prep_dir.mkdir(parents=True, exist_ok=True)
        output_ms = prep_dir / "selected.ms"
        marker = prep_dir / "complete.json"
        if marker.exists() and output_ms.exists() and not overwrite:
            print(f"Skipping completed selection {project}/{variant}")
            continue

        available, omitted_vis = resolve_available_config(config, data_dir)
        vis = list(available["vis"])
        selections = {
            key: as_list(available.get(key, ""), len(vis))
            for key in ("field", "spw", "antenna", "scan", "intent")
        }
        parts = []
        for index, vis_path in enumerate(vis):
            part = prep_dir / f"selected_{index}.ms"
            _remove_if_requested(part, overwrite)
            mstransform(
                vis=vis_path,
                outputvis=str(part),
                field=selections["field"][index],
                spw=selections["spw"][index],
                antenna=selections["antenna"][index],
                scan=selections["scan"][index],
                intent=selections["intent"][index],
                datacolumn=available.get("datacolumn", "data"),
                reindex=True,
            )
            parts.append(part)

        _remove_if_requested(output_ms, overwrite)
        if len(parts) == 1:
            shutil.move(str(parts[0]), str(output_ms))
        else:
            concat(vis=[str(path) for path in parts], concatvis=str(output_ms))

        write_manifest(
            marker,
            {
                "project": project,
                "variant": variant,
                "source_measurement_sets": vis,
                "omitted_unavailable_measurement_sets": omitted_vis,
                "selected_measurement_set": str(output_ms),
                "selection": selections,
                "source_datacolumn": available.get("datacolumn", "data"),
                "selected_datacolumn": "data",
            },
        )


def main() -> None:
    """Parse command-line arguments and prepare selected Measurement Sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=(*PROJECTS, "all"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    projects = PROJECTS if args.project == "all" else (args.project,)
    for project in projects:
        prepare_project(project, args.overwrite)


if __name__ == "__main__":
    main()
