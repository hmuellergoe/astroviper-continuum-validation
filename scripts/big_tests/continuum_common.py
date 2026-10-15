"""Shared configuration helpers for the continuum validation runs."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "big_tests"
RESULTS_ROOT = ROOT / "results" / "big_tests"

PROJECTS = (
    "2017.1.00661.S",
    "2017.1.00999.S",
    "2019.1.00678.S",
    "2023.1.01342.S",
)

BIG_VALIDATION_PROJECTS = (
    "2017.1.00661.S",
    "2023.1.01342.S",
)

ASTROVIPER_DECONVOLVERS = ("hogbom", "mtmfs")


def _balanced_tclean_calls(source: str) -> list[str]:
    """Extract complete ``tclean(...)`` expressions from a CASA command file."""
    calls: list[str] = []
    position = 0
    while (start := source.find("tclean(", position)) >= 0:
        depth = 0
        quote = None
        escaped = False
        end = None
        for index, character in enumerate(source[start + 6 :], start + 6):
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
            elif character in "'\"":
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise ValueError("Unterminated tclean call in command file")
        calls.append(source[start:end])
        position = end
    return calls


def continuum_configs(project: str) -> list[dict[str, Any]]:
    """Return literal keyword arguments for every MFS call in one project."""
    if project not in PROJECTS:
        raise ValueError(f"Unknown project {project!r}; choose from {PROJECTS}")
    paths = sorted((DATA_ROOT / project).glob("tclean_cmd*.py"))
    paths = [path for path in paths if not path.name.endswith("~")]
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected one tclean command file for {project}, got {paths}"
        )

    configs = []
    for expression in _balanced_tclean_calls(paths[0].read_text()):
        call = ast.parse(expression).body[0].value
        keywords = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
        if str(keywords.get("specmode", "mfs")).lower() == "mfs":
            configs.append(keywords)
    return configs


def variant_name(index: int, config: dict[str, Any]) -> str:
    """Return a stable short name for one continuum configuration."""
    spws = config.get("spw", [])
    spw_ids = re.findall(r"(?:^|,)(\d+)(?=:|,|$)", str(spws[0])) if spws else []
    suffix = "_".join(spw_ids) if spw_ids else str(index)
    return f"mfs_{index}_{suffix}"


def project_variants(project: str) -> list[tuple[str, dict[str, Any]]]:
    """Pair each continuum configuration with its result-directory name."""
    return [
        (variant_name(index, config), config)
        for index, config in enumerate(continuum_configs(project))
    ]


def astroviper_variants(
    project: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Expand each archived configuration into Hogbom and two-term MT-MFS runs."""
    variants = []
    for variant, source_config in project_variants(project):
        for deconvolver in ASTROVIPER_DECONVOLVERS:
            config = dict(source_config)
            config.update(deconvolver=deconvolver, nterms=2)
            variants.append((variant, deconvolver, config))
    return variants


def as_list(value: Any, length: int) -> list[Any]:
    """Broadcast a CASA scalar or one-element list to an MS list length."""
    if isinstance(value, (list, tuple)):
        if len(value) == length:
            return list(value)
        if len(value) == 1:
            return list(value) * length
        raise ValueError(f"Cannot broadcast {value!r} to {length} measurement sets")
    return [value] * length


def resolve_available_config(
    config: dict[str, Any], data_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    """Resolve archived MS names and retain only inputs present in the test archive."""
    archived_vis = list(config["vis"])
    requested_datacolumn = str(config.get("datacolumn", "data"))
    resolved: list[tuple[int, Path, str]] = []
    missing = []
    for index, name in enumerate(archived_vis):
        candidates = [(data_dir / name, requested_datacolumn)]
        if name.endswith("_targets.ms"):
            candidates.append(
                (data_dir / name.replace("_targets.ms", ".ms"), "corrected")
            )
        if name.endswith("_targets_line.ms"):
            candidates.append(
                (data_dir / name.replace("_targets_line.ms", ".ms"), "corrected")
            )
        match = next(
            ((path, datacolumn) for path, datacolumn in candidates if path.exists()),
            None,
        )
        if match is None:
            missing.append(name)
        else:
            path, datacolumn = match
            resolved.append((index, path, datacolumn))
    if not resolved:
        raise FileNotFoundError(
            f"None of the archived measurement sets exist beneath {data_dir}: "
            f"{archived_vis}"
        )

    available = dict(config)
    available["vis"] = [str(path) for _, path, _ in resolved]
    datacolumns = {datacolumn for _, _, datacolumn in resolved}
    if len(datacolumns) != 1:
        raise ValueError(
            "Resolved inputs require mixed data columns, which CASA cannot select "
            f"in one tclean call: {sorted(datacolumns)}"
        )
    available["datacolumn"] = datacolumns.pop()
    for key in ("field", "spw", "antenna", "scan", "intent"):
        value = config.get(key, "")
        if isinstance(value, (list, tuple)) and len(value) == len(archived_vis):
            available[key] = [value[index] for index, _, _ in resolved]
    return available, missing


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a reproducibility manifest beneath a result directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def parse_jy(value: Any) -> float:
    """Convert a CASA Jy or mJy threshold to Jy."""
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(m?Jy)\s*",
        str(value),
    )
    if match is None:
        raise ValueError(f"Unsupported threshold {value!r}")
    scale = 1e-3 if match.group(2) == "mJy" else 1.0
    return float(match.group(1)) * scale


def parse_cell_radians(value: Any) -> float:
    """Convert the supported CASA angular cell notation to radians."""
    cell = value[0] if isinstance(value, (list, tuple)) else value
    match = re.fullmatch(r"\s*([0-9.eE+-]+)\s*(arcsec|arcmin|deg)\s*", str(cell))
    if match is None:
        raise ValueError(f"Unsupported cell size {cell!r}")
    factors = {"arcsec": 3600.0, "arcmin": 60.0, "deg": 1.0}
    return float(match.group(1)) * 3.141592653589793 / (180.0 * factors[match.group(2)])
