# AstroVIPER continuum unit-suite snapshot

This directory is a self-contained harness copy of the focused continuum unit
tests from AstroVIPER commit `1f5807e` on branch `RADPS-roadmap-194`. It retains
the processing-function, node-task, and distributed-application directory
structure so failures identify the affected application layer.

The copied session fixture uses the harness's local five-channel TW Hydra
processing set when it is present. In a scripts-only clone without `data/`, it
falls back to downloading the same published fixture used by the AstroVIPER
repository.

Run only this snapshot from the harness root with:

```bash
$VALIDATION_PYTHON -m pytest -q scripts/light_tests/repository_unit_suite
```

The files are a versioned snapshot rather than symlinks. When the upstream
continuum tests change, refresh this directory deliberately and update the
source commit above so the harness records exactly which contracts it tests.
