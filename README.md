# AstroVIPER continuum validation

This repository contains the executable and retained evaluation portions of
the AstroVIPER continuum-imaging validation harness. Large Measurement Sets,
processing sets, and generated reconstruction products are deliberately not
versioned; place them under `data/` and `results/` using the layout documented
in [`TESTS.md`](TESTS.md).

The harness uses a category-first layout. Each category is subdivided into
`light_tests`, `medium_tests`, and `big_tests`:

- `scripts/` contains executable tests, CASA/AstroVIPER runners, and helper
  modules;
- `data/` contains source Measurement Sets and processing sets;
- `results/` contains generated images, neutral comparison products, work
  caches, and retained CASA products;
- `evaluation/` contains comparison code, metrics, reports, and plots.

The test-size tiers retain their original meaning: light tests use synthetic
arrays, medium tests use compact CASA and TW Hydra regression datasets, and big
tests perform archived ALMA reconstructions.

Create a Python environment containing AstroVIPER and its dependencies, then
define these shell variables before using the documented commands:

```bash
export VALIDATION_PYTHON=/path/to/python
export ASTROVIPER_SOURCE=/path/to/astroviper/src
export CASA_PYTHON=/path/to/casa/python
```

CASA runs additionally accept `CASA_DATA_PATH` and `CASA_TEST_DATA_PATH`; both
default to `~/.casa/data` when omitted.

See [`TESTS.md`](TESTS.md) for the coverage matrix, commands, expected
differences, and the relationship between the three tiers.
