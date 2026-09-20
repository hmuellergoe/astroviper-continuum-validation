# Continuum imaging validation

This directory validates AstroVIPER continuum imaging at three complementary
scales. A failure in a lower tier normally identifies a mathematical or state
management error; a failure only in a higher tier more often indicates an I/O,
selection, orchestration, numerical-agreement, or deconvolution issue.

## Test tiers

| Tier | Location | Typical cost | Purpose |
|---|---|---:|---|
| Light | `scripts/light_tests/` | seconds | Synthetic tests of processing functions, node state, and distributed reducers |
| Medium | `scripts/medium_tests/` | minutes | Targeted CASA comparisons plus TW Hydra distributed continuum and cube regressions |
| Big | `scripts/big_tests/` | minutes to hours; large storage | Full archived ALMA `tclean` reconstructions and image-level comparisons |
| Repository | `astroviper/tests/unit/*/imaging/test_continuum.py` plus `test_iteration_control.py` | seconds | CI coverage across processing, node-task, and distributed continuum layers |

## Lightweight synthetic tests

Run from `continuum/` with the AstroVIPER checkout on `PYTHONPATH`:

```bash
PYTHONPATH=$ASTROVIPER_SOURCE \
  $VALIDATION_PYTHON -m pytest -q scripts/light_tests
```

The tests mirror AstroVIPER's application layers.

### Distributed reduction

`test_mvc_frequency_reduction.py` verifies that:

- MVC channel images have exclusive frequency ownership, are sorted, and are
  concatenated without accidental summation;
- duplicate MVC channel ownership is rejected;
- task-local MVC observed-visibility grids survive tree reduction without being
  combined across frequency tasks;
- the MFS reference-frequency primary beam is retained once when identical,
  accepted when supplied by one leaf, and rejected when inconsistent.

### Node-task state

`test_continuum_node_state.py` verifies that:

- final static products replace an unused chunk-frequency coordinate with the
  single reference-frequency coordinate;
- replacement is rejected while a channel cube still uses the old coordinate;
- a single reduce leaf is normalized into task-keyed PB and weighting caches;
- task-local MVC observed-grid mappings survive the append node for reuse by the
  next distributed graph.

### Processing mathematics

- `test_mvc_normal_equations.py`: weighted MVC Taylor residuals, PSF/Hessian
  orders, PB-limit masking, and the channel-PB/effective-PB prediction rule.
- `test_mfs_primary_beam.py`: MFS PB evaluation at the configured reference
  frequency and removal of channel PB accumulators.
- `test_grid_imaging_weights.py`: CASA-compatible UV-cell truncation and
  matching Briggs density degridding.
- `test_continuum_restoration.py`: PB correction uses restored Taylor term zero,
  masks below `pblimit`, registers its output group, and rejects a channelized PB.

## Medium CASA regression tests

`data/medium_tests/` contains the small CASA regression datasets `refim_point.ms` and
`refim_oneshiftpoint.mosaic.ms`. It compares neutral NPZ products rather than
requiring CASA and AstroVIPER in one Python environment.

Coverage includes:

- MFS and MVC residual Taylor terms 0 and 1;
- PSF/Hessian Taylor orders 0, 1, and 2;
- effective/reference primary beams;
- natural-local, Briggs-local, and Briggs-global weighting;
- persisted Zarr products reopened from `image_store` and compared with the
  in-memory application result before NPZ export.
- a shared 0.1 Jy threshold-cleaning regression requiring nonzero CASA and
  AstroVIPER Taylor-zero models and lower residual peaks.

`scripts/medium_tests/cube_backward_compatibility/` uses the five-channel TW Hydra
processing set for several distributed-application checks. The cube test reopens a
newly persisted dirty cube and compares its dimensions, coordinates, residual,
PSF, primary beam, and fitted PSF beam parameters against a frozen AstroVIPER
baseline. The continuum test runs the public application with one and five
frequency partitions and requires equivalent persisted residual, PSF, and PB
products for MFS natural-local, MVC natural-local, and MFS Briggs-global. All
comparisons use a peak relative tolerance of `1e-6`. Real-cleaning MFS and MVC
cases additionally compare `visibility_memory_mode="recompute"`, `"in_memory"`,
and disk-backed `"in_place"` through multiple residual/model-update cycles.

The partition-invariance cases intentionally disable cleaning so they isolate
major-cycle behavior. Separate repository TW Hydra cases exercise real MFS and
MVC cleaning, restoration, PB correction, later-cycle prediction, additional
weighting schemes, automatic chunking, single precision, and direct disk I/O.

Run AstroVIPER and compare existing CASA references:

```bash
cd /path/to/astroviper-continuum-validation
PYTHONPATH=$ASTROVIPER_SOURCE \
  $VALIDATION_PYTHON scripts/medium_tests/run_radps_reference.py \
  --cases phase_center --modes mfs mvc --n-chunks 2
$VALIDATION_PYTHON evaluation/medium_tests/compare_results.py \
  --cases phase_center --modes mfs mvc
```

Run the shared-threshold cleaning regression:

```bash
cd /path/to/astroviper-continuum-validation
$CASA_PYTHON \
  scripts/medium_tests/run_casa_reference.py --cases phase_center --modes mfs \
  --weightings natural_local --clean
PYTHONPATH=$ASTROVIPER_SOURCE \
  $VALIDATION_PYTHON scripts/medium_tests/run_radps_reference.py --cases phase_center --modes mfs \
  --weightings natural_local --clean
$VALIDATION_PYTHON evaluation/medium_tests/verify_cleaning.py
```

Run the TW Hydra distributed continuum and cube guards with the shared imaging
environment:

```bash
cd /path/to/astroviper-continuum-validation
PYTHONPATH=$ASTROVIPER_SOURCE \
  $VALIDATION_PYTHON scripts/medium_tests/cube_backward_compatibility/test_tw_hydra_cube.py
```

Briggs-local intentionally uses a different density convention from CASA's
global continuum weighting and is reported as `EXPECTED-DIFFERENCE`. Enforced
natural and Briggs-global products must pass.

## Large reconstruction tests

`data/big_tests/` contains archived `tclean` calls for four ALMA projects and five MFS
configurations. It applies the same field, SPW/channel, antenna, scan, intent,
and data-column selections, then compares CASA FITS products with AstroVIPER
Zarr products.

Products include residual, PSF, primary beam, model, restored image, and—when
requested—the PB-corrected restored image. The evaluator reports finite-pixel
status, correlation, normalized RMSE, flux ratio, orientation, and diagnostic
plots. Unlike the medium tier, these runs exercise restoration and the configured
minor cycle, subject to the documented CASA auto-multithresh versus AstroVIPER
Hogbom difference.

Run from the `continuum` harness root using the shared environment:

```bash
cd /path/to/astroviper-continuum-validation
$VALIDATION_PYTHON scripts/big_tests/run_astroviper.py all --cores 4 --memory-limit 8GB
$VALIDATION_PYTHON evaluation/big_tests/evaluate_reconstructions.py all
```

Use one project or `--niter 0` for a smoke test before running every large case.

## Repository regression test

The AstroVIPER repository keeps continuum tests at the processing-function,
node-task, and distributed-application layers. They cover Taylor normal
equations, weighting, deconvolution state, restoration, output filtering and
Zarr persistence, plus real-data TW Hydra execution paths. The persisted-output
checks guard against the former NaN-placeholder defect.

## Demonstrator evaluator tests

The repository also carries the portable continuum-demonstrator evaluation
programs at the top level of `evaluation/`. Their lightweight QA2 and result
discovery tests run without the excluded Measurement Sets or image stores:

```bash
$VALIDATION_PYTHON -m pytest -q evaluation/test_evaluate_science_metrics.py
```

## Current intentional limitations

- The default medium CASA matrix isolates the major cycle; its shared-threshold
  case and the repository TW Hydra tests validate deconvolution separately.
- The big tier currently contains archived MFS calls; MVC is covered by the light
  and medium tiers.
- CASA auto-multithresh and AstroVIPER Hogbom do not implement the same minor-loop
  masking strategy, so deep-clean image agreement is not expected to be exact.
- Briggs-local is informational; Briggs-global is the CASA-equivalent continuum
  weighting comparison.

## Verification status

Last verified on 2026-08-24 against the `RADPS-roadmap-194` working tree:

- lightweight suite: 15 passed;
- focused AstroVIPER continuum unit suite: 187 passed;
- medium phase-center matrix: all six MFS/MVC × natural-local/Briggs-local/
  Briggs-global runs completed, their persisted products matched the in-memory
  results, and every enforced CASA comparison passed;
- TW Hydra distributed continuum regression: MFS natural-local, MVC
  natural-local, and MFS Briggs-global passed with equivalent persisted results
  for one and five frequency partitions;
- TW Hydra five-channel cube backward-compatibility regression: persisted
  residual, PSF, and primary-beam image arrays match; the frozen fitted-beam
  metadata differs by about 4% because the current shared fitter uses the
  corrected pixel scale, so that metadata baseline must be refreshed;
- big 2017.1.00661.S and 2023.1.01342.S reconstructions: CASA and AstroVIPER
  produce finite, nonzero models and all six reconstruction products are
  compared. The 2017 residual/model correlations are `0.999978`/`0.999999`; the
  2023 residual/model correlations are `0.991862`/`0.967543`.

The audit fixed two defects: PB correction now emits only restored Taylor term
zero, and the worker-side forward FFT no longer calls an interactive
`matplotlib.pyplot.show()`. The latter reduced the affected medium cases from
roughly 15 minutes of blocking to about 4–5 seconds of driver execution.
