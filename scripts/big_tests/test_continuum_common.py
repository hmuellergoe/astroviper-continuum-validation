"""Tests for archived measurement-set fallback resolution."""

from pathlib import Path

from continuum_common import (
    BIG_VALIDATION_PROJECTS,
    astroviper_variants,
    resolve_available_config,
)
from scripts.iteration_limits import casa_to_astroviper_iteration_limit


def test_casa_iteration_limits_translate_to_astroviper_budgets():
    """Positive CASA limits are inclusive; zero and unlimited stay unchanged."""
    assert casa_to_astroviper_iteration_limit(0) == 0
    assert casa_to_astroviper_iteration_limit(5) == 6
    assert casa_to_astroviper_iteration_limit(100) == 101
    assert casa_to_astroviper_iteration_limit(-1) == -1


def test_targets_fallback_uses_calibrated_column(tmp_path: Path):
    """A restored full MS must supply CORRECTED_DATA for a missing target MS."""
    (tmp_path / "observation.ms").mkdir()
    config = {"vis": ["observation_targets.ms"], "datacolumn": "data"}

    available, missing = resolve_available_config(config, tmp_path)

    assert missing == []
    assert available["vis"] == [str(tmp_path / "observation.ms")]
    assert available["datacolumn"] == "corrected"


def test_existing_targets_ms_keeps_requested_column(tmp_path: Path):
    """A real pipeline target MS retains its configured DATA selection."""
    (tmp_path / "observation_targets.ms").mkdir()
    config = {"vis": ["observation_targets.ms"], "datacolumn": "data"}

    available, missing = resolve_available_config(config, tmp_path)

    assert missing == []
    assert available["vis"] == [str(tmp_path / "observation_targets.ms")]
    assert available["datacolumn"] == "data"


def test_selected_big_cases_expand_to_hogbom_and_two_term_mtmfs():
    """Every archived configuration gets both requested deconvolution runs."""
    matrix = []
    for project in BIG_VALIDATION_PROJECTS:
        variants = astroviper_variants(project)
        matrix.extend(
            (project, variant, deconvolver)
            for variant, deconvolver, _ in variants
        )

        assert len(variants) == 2
        assert [deconvolver for _, deconvolver, _ in variants] == [
            "hogbom",
            "mtmfs",
        ]
        assert all(config["nterms"] == 2 for _, _, config in variants)
    assert len(matrix) == 4
