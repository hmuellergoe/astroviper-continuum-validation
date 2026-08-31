"""Tests for archived measurement-set fallback resolution."""

from pathlib import Path

from continuum_common import resolve_available_config


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
