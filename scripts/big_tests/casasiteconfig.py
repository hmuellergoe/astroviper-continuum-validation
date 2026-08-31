"""Portable CASA data configuration for the continuum validation harness."""

import os
from pathlib import Path


_casa_data = Path(
    os.environ.get("CASA_DATA_PATH", Path.home() / ".casa" / "data")
).expanduser()
_casa_test_data = Path(
    os.environ.get("CASA_TEST_DATA_PATH", _casa_data)
).expanduser()

measurespath = str(_casa_data)
datapath = [str(_casa_test_data), str(_casa_data)]
measures_auto_update = False
data_auto_update = False
