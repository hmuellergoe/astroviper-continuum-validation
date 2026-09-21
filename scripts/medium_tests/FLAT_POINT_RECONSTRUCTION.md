# Flat-spectrum phase-centre reconstruction

This example is part of the controlled point-source results in Memo II, IV-A.1.
It is a first matched single-term Högbom reconstruction test, not a replacement
for the two-term dirty-image or off-axis harness cases.

## Input and controls

`prepare_flat_point.py` copies `refim_point.ms`, requires one field and only
parallel-hand correlations, and sets DATA and any CORRECTED_DATA to the exact
visibility of a one-Jy unpolarized point at the phase centre: 1+0j at every
frequency and baseline. Sampling, weights, flags and coordinates are retained;
no noise or spectral slope is injected. Scratch model state is removed. The
original dataset is untouched, and the preparation validates the unit data.

Both implementations use MFS, nterms=1, Högbom, natural weighting, a 200×200
image, 10 arcsec pixels, 1.5 GHz reference frequency, gain 0.1, a requested
five-iteration minor-cycle cap, and a radius-three-pixel mask at (100,100).
AstroVIPER receives the mask as an NPY file; CASA receives the corresponding
pixel-region expression. The dirty stage suppresses cleaning; the update stage
sets a five-iteration total cap; the clean stage allows 100 iterations and 20
major cycles with a 0.01 Jy threshold. Each stage starts independently from a
zero model. Native restoration and PB correction are enabled. Common-beam
restoration is a subsequent extension, not implemented here.

## Run

From the repository root, set `PY` to the Python executable in the viper
(or equivalent CASA/AstroVIPER) environment. Use a fresh output directory;
existing engine outputs are rejected to prevent accidental CASA restarts.

```bash
PY=/export/home/smurf/anaconda3/envs/viper/bin/python
$PY scripts/medium_tests/prepare_flat_point.py
$PY scripts/medium_tests/run_flat_point_reconstruction.py --engine casa
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY scripts/medium_tests/run_flat_point_reconstruction.py --engine astroviper
$PY evaluation/medium_tests/verify_flat_point_reconstruction.py
```

Both runners accept `--ms` and `--output-dir`; the evaluator accepts `--results`.
Defaults use data/medium_tests/refim_flat_point.ms and
results/medium_tests/flat_point. Preparation accepts --input and --output and
refuses to replace existing inputs. Scripts and this description belong in Git;
generated Measurement Sets, images and logs remain under ignored data/results.

## Evaluation

The evaluator requires the same 0.05 relative / 0.005 absolute whole-array
criteria as the existing point-source harness for model, residual, PSF, PB,
restored and PB-corrected products at each stage. It explicitly exports Stokes I
from AstroVIPER rather than comparing incidental additional polarization planes.
For the known one-pixel model it independently checks the linear prediction
identity R=(1−M_center)R_dirty, while requiring no model flux outside the source
pixel. It requires dirty peak agreement with one Jy within 0.001, final restored
peak agreement within 0.01, and final residual peak no larger than 0.01 Jy.
All results, including failures, are saved in checks.json; a failed check returns
nonzero. Returned iteration/cycle descriptions and fitted beams are also saved.
The fixed-update arrays need not pass merely because final thresholded outputs
do. Iteration-count and mask semantics require inspection before asserting
identical controller behavior. This example does not yet assess integrated-flux
accuracy, common-restoring-beam equivalence or global Briggs reconstruction.

## First recorded run

AstroVIPER 795b8ba (0.0.37), CASA 6.7.7.6: dirty and final-clean comparisons pass, but the fixed-update model/residual comparisons fail by about 0.059049 Jy. Both final residual peaks are approximately 0.0096977 Jy and restored peaks approximately one Jy/beam. Prediction checks pass independently for each model. The complete evaluator therefore correctly exits 1 (28/30 checks pass). See results/flat_point_2026-09-20/REPORT.md and final/checks.json.

Returned iteration counts differ and require interpretation. AstroVIPER also reports masksum=40000 despite the supplied 29-pixel mask; this one-source result does not validate mask enforcement. These limitations remain open; no full reconstruction/controller equivalence claim is made.

## Follow-up with CASA counting restored

At the user's request, a local AstroVIPER patch on 795b8ba restores the positive inclusive Högbom iteration limit with a capped reported counter. Both kernels were rebuilt in viper; 59 unit tests pass. Rerunning the three AstroVIPER stages against the same CASA references now passes all 30 example checks, including the previously failing fixed-update comparisons. See results/flat_point_casa_counting/REPORT.md and astroviper.patch. The initial failure above remains a historical result of the earlier exclusive-limit build. Other controller, mask-summary and coverage caveats remain.
