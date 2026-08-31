# CASA versus RADPS continuum major-cycle validation

This directory contains a deliberately small, single-field validation harness.
It compares CASA `tclean` with the current RADPS/AstroVIPER continuum major
cycle without claiming to validate RADPS's temporary Taylor-zero Högbom minor
loop.

It also contains `cube_backward_compatibility/`, which uses a five-channel TW
Hydra processing set to exercise the public distributed continuum application
and protect the established cube path. The continuum cases compare one versus
five frequency partitions for MFS natural-local, MVC natural-local, and MFS
Briggs-global imaging. The cube case compares a regenerated dirty image with a
frozen AstroVIPER baseline. Every comparison reopens its Zarr output and uses a
`1e-6` peak relative tolerance.

## Selected CASA tests and data

The primary source is CASA's
`casatasks/tests/tasks/test_task_tclean.py`:

- `test_mtmfsviacube_compare_with_mfs` uses `refim_point.ms` to compare MFS and
  MVC for a phase-center point source. This is the best basic equivalence test.
- `test_mtmfsviacube_standard` uses field 0 of
  `refim_oneshiftpoint.mosaic.ms` with the standard gridder. Despite the MS
  filename, this is used here as a single selected field and provides an
  off-axis PB-sensitive MVC case.

Copied data under `data/medium_tests/`:

- `refim_point.ms` (required basic case)
- `refim_oneshiftpoint.mosaic.ms` (optional off-axis case)

## Why the minor loop is thresholded off

CASA uses the real coupled MT-MFS minor loop. RADPS currently adapts Taylor term
zero to a cube Högbom minor loop. CASA uses `niter=0`. The current RADPS driver
still dispatches its minor wrapper for `niter=0`, so the RADPS harness uses one
allowed iteration with threshold `1e30`; the model remains unchanged. The
harness compares these major-cycle products:

- residual Taylor terms 0 and 1;
- PSF/Hessian Taylor orders 0, 1, and 2;
- primary-beam Taylor/effective plane.

## Threshold-cleaning regression

The optional `--clean` case uses the same phase-center point source in CASA and
AstroVIPER with a shared `0.1 Jy` stopping threshold, 100-iteration cap, and
natural weighting. Its dirty peak is approximately `1.06 Jy/beam`, so the
threshold must activate CLEAN while remaining far above the image noise. Clean
outputs use the `_clean` suffix and never replace the major-cycle references.

This regression deliberately checks the common behavior supported by both
implementations: Taylor-zero model structure must be nonzero and the residual
peak must decrease. It does not require the CASA MT-MFS model to numerically
match AstroVIPER's current Taylor-zero Högbom adaptation.

Natural weighting is used first because local and global natural weighting are
identical and it isolates MFS/MVC mathematics from weight-density differences.

The harness also defines `briggs_local` and `briggs_global`, both with CASA
compatibility enabled and `robust=0.5`. CASA supplies the same globally reduced
Briggs reference for both. RADPS local retains one density plane and Briggs
factor per channel. RADPS global reduces all partitions and then collapses all
selected channels into one common continuum density plane and factor, matching
CASA. The local comparison is therefore informational and is not expected to
match the CASA continuum reference.

## Run

Run CASA references with a CASA Python executable:

```bash
cd /path/to/astroviper-continuum-validation
$CASA_PYTHON \
  scripts/medium_tests/run_casa_reference.py \
  --cases phase_center --modes mfs mvc
```

Run RADPS in the development environment containing AstroVIPER, GraphVIPER,
ToolVIPER, and xradio. The intended command is:

```bash
$VALIDATION_PYTHON scripts/medium_tests/run_radps_reference.py \
  --cases phase_center --modes mfs mvc --overwrite-conversion
```

Compare:

```bash
$VALIDATION_PYTHON evaluation/medium_tests/compare_results.py \
  --cases phase_center --modes mfs mvc
```

Run the threshold-cleaning regression:

```bash
$CASA_PYTHON \
  scripts/medium_tests/run_casa_reference.py \
  --cases phase_center --modes mfs --weightings natural_local --clean
$VALIDATION_PYTHON scripts/medium_tests/run_radps_reference.py \
  --cases phase_center --modes mfs --weightings natural_local --clean
$VALIDATION_PYTHON evaluation/medium_tests/verify_cleaning.py
```

Run the TW Hydra distributed continuum and cube tests:

```bash
PYTHONPATH=$ASTROVIPER_SOURCE \
  $VALIDATION_PYTHON \
  scripts/medium_tests/cube_backward_compatibility/test_tw_hydra_cube.py
```

The comparison command includes `natural_local`, `briggs_local`, and
`briggs_global` by default. Use `--weightings` only when selecting a subset.

Run the added Briggs/local-versus-global tests:

```bash
$CASA_PYTHON \
  scripts/medium_tests/run_casa_reference.py \
  --cases phase_center --modes mfs mvc \
  --weightings briggs_local briggs_global
$VALIDATION_PYTHON scripts/medium_tests/run_radps_reference.py \
  --cases phase_center --modes mfs mvc --n-chunks 2 \
  --weightings briggs_local briggs_global
$VALIDATION_PYTHON evaluation/medium_tests/compare_results.py \
  --cases phase_center --modes mfs mvc \
  --weightings briggs_local briggs_global
```

After the basic case works, add the PB-sensitive case:

```bash
$CASA_PYTHON \
  scripts/medium_tests/run_casa_reference.py \
  --cases off_axis --modes mvc
$VALIDATION_PYTHON scripts/medium_tests/run_radps_reference.py \
  --cases off_axis --modes mvc --overwrite-conversion
$VALIDATION_PYTHON evaluation/medium_tests/compare_results.py \
  --cases off_axis --modes mvc
```

The comparison searches the finite set of transpose/axis-flip conventions once
for each result pair, reports the chosen orientation, and then uses that same
orientation for every product. This makes axis-convention differences visible
without allowing a different transform to hide errors in individual products.

The scripts explicitly select CASA field `0` and the corresponding xradio field
name (`fake` or `fake0`). The off-axis test also applies the same explicit J2000
phase center on both sides; it does not accidentally combine the two fields in
the source MS.

## Validation performed here

Both CASA phase-center references were executed successfully with CASA 6.7.7.3.
They are saved under `results/medium_tests/`. Use the interpreter selected by
`VALIDATION_PYTHON` for all AstroVIPER and evaluation commands.

This legacy CASA dataset records its scan intent as the literal string
`"None"`. Requesting `OBSERVE_TARGET#ON_SOURCE` empties the metadata Processing
Set and must not be used for this case.

The Briggs local/global phase-center matrix has also been executed for both MFS
and MVC. All four runs complete. For global Briggs (`robust=0.5`), every MFS and
MVC residual and PSF Taylor product passes. MVC PB also passes. MFS PB retains
the same independent effective-PB discrepancy seen with natural weighting.
