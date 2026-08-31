# CASA/RADPS continuum validation status

The validated runtime fixes have been ported to
`$ASTROVIPER_CHECKOUT`; no temporary source mirror is required.

## Completed

- CASA phase-center MFS and MVC reference products generated.
- RADPS phase-center MFS and MVC completed with two frequency chunks.
- MVC passes for residual Taylor terms, PSF Taylor orders, and primary beam.
- MFS passes for residual Taylor terms and PSF Taylor orders.
- MFS PB remains outside tolerance (RMS `0.01555`, maximum `0.04373`) because
  its effective-PB convention still differs from CASA.
- Added `briggs_local` and `briggs_global` (`robust=0.5`) variants for CASA and
  RADPS, for both MFS and MVC. All four RADPS variants run to completion.
- Global Briggs now collapses the reduced physical-frequency planes into one
  common continuum density grid and Briggs factor, matching CASA. Local Briggs
  deliberately retains channel-dependent density planes and has no direct CASA
  continuum equivalent.
- Relative to CASA global Briggs, every MFS and MVC residual and PSF Taylor
  product passes. MVC PB passes. MFS PB retains the same independent discrepancy
  seen under natural weighting.

## Run

Install AstroVIPER editable in `viper`, then:

```bash
cd /path/to/astroviper-continuum-validation
$VALIDATION_PYTHON scripts/medium_tests/run_radps_reference.py \
  --cases phase_center --modes mfs mvc --n-chunks 2
$VALIDATION_PYTHON evaluation/medium_tests/compare_results.py \
  --cases phase_center --modes mfs mvc
```

For Briggs local/global:

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

This legacy CASA dataset records its scan intent as the literal string
`"None"`; the harness selects that value intentionally.
