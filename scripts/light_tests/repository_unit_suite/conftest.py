"""Shared TW Hydra input for the copied AstroVIPER continuum unit suite."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import gdown
import pytest


@pytest.fixture(scope="session")
def tw_hydra_archive(tmp_path_factory):
    """Archive the harness fixture locally, falling back to its public download."""
    with TemporaryDirectory(
        prefix="tw_hydra_download_", dir=tmp_path_factory.getbasetemp()
    ) as directory:
        archive = Path(directory) / "twhya_selfcal_lsrk_5chans.ps.zarr.zip"
        harness_root = Path(__file__).resolve().parents[3]
        local_store = (
            harness_root
            / "data"
            / "medium_tests"
            / "cube_backward_compatibility"
            / "twhya_selfcal_lsrk_5chans.ps.zarr"
        )
        if local_store.is_dir():
            shutil.make_archive(
                str(archive.with_suffix("")),
                "zip",
                root_dir=local_store.parent,
                base_dir=local_store.name,
            )
        else:
            result = gdown.download(
                id="1BRe3cD6YAWkn-jSPbClGGM9VlbxHP_yn",
                output=str(archive),
                quiet=True,
            )
            if result is None or not archive.is_file():
                raise RuntimeError(
                    "Could not obtain the shared TW Hydra reference data"
                )
        yield archive
