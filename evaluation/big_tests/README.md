# Continuum validation

This directory contains comparisons between the archived CASA `tclean` continuum
commands and AstroViper's `image_continuum_single_field` implementation. Generated
metrics, plots, and summaries remain outside the AstroViper repository.

Each comparison reports pixel correlation, RMSE, normalized RMSE, mean difference,
RMS, peak magnitude, and integrated-flux ratio. The evaluator checks transpose and
axis-flip conventions and records the selected orientation in `metrics.json`; it
does not resample either reconstruction.

When both implementations produced it, the comparison includes the
primary-beam-corrected restored image (`image.pbcor` / `SKY_RESTORED_PBCOR`).

Run all completed comparisons with:

```bash
$VALIDATION_PYTHON evaluation/big_tests/evaluate_reconstructions.py all
```

The outputs are `summary.csv`, `summary.md`, a generated `finding.md`, and one
three-panel comparison plot per available product under `<project>/<variant>/`.
Running the evaluator for one project refreshes that project's products while
retaining every other completed project in the global summaries.

## Known input constraint

Some archived commands refer to pipeline-produced `_targets.ms` datasets that
are not present in the validation archive. When the harness falls back to a full
measurement set, it selects calibrated `CORRECTED_DATA`; the preparation step
places those values in the selected measurement set's `DATA` column. Using the
full measurement set's raw `DATA` column can lower amplitudes by orders of
magnitude and incorrectly leave both CASA and AstroViper models empty.
