# Continuum verification scripts

The four data directories contain five archived continuum configurations:
project `2019.1.00678.S` has both an SPW-7 MFS call and a four-SPW continuum
call. The routine big validation matrix selects `2017.1.00661.S` and
`2023.1.01342.S`; passing `all` runs both deconvolvers for those two examples,
for four runs total. The other archived projects remain runnable by name.

Run the commands below from the repository root after defining
`VALIDATION_PYTHON` and `CASA_PYTHON` as described in the top-level README.

## Workflow

1. Run the archived CASA configurations from scratch and export FITS products:

   ```bash
   export CASASITECONFIG="$PWD/scripts/big_tests/casasiteconfig.py"
   export MPLCONFIGDIR="$PWD/results/big_tests/.matplotlib"
   "$CASA_PYTHON" scripts/big_tests/run_casa_reference.py all
   ```

2. Apply the same field/SPW/channel/antenna/scan/intent/data-column selections and
   concatenate multi-execution-block inputs into one selected MS per configuration:

   ```bash
   "$VALIDATION_PYTHON" scripts/big_tests/prepare_selected_ms.py all
   ```

3. Use a Python 3.12 environment containing CASA tools, XRADIO, GraphVIPER,
   ToolVIPER, and this AstroViper checkout. Convert the selected MS and image
   every configuration twice: once with Hogbom and once with MT-MFS, both with
   `nterms=2`:

   ```bash
   "$VALIDATION_PYTHON" scripts/big_tests/run_astroviper.py all --cores 4 --memory-limit 8GB
   ```

   Positive CASA `niter` and `cycleniter` limits are inclusive and are
   translated to AstroVIPER budgets by adding one. Zero and negative/unlimited
   limits are preserved.

   Four dataset-specific entry points are also provided. For example:

   ```bash
   "$VALIDATION_PYTHON" scripts/big_tests/run_astroviper_2023_1_01342_S.py --cores 4
   ```

4. Evaluate every completed pair:

   ```bash
   "$VALIDATION_PYTHON" evaluation/big_tests/evaluate_reconstructions.py all
   ```

The shared selected processing set lives below
`results/big_tests/<project>/<variant>/astroviper/`. Each image is written to
its `hogbom/` or `mtmfs/` child. Evaluation reports use the corresponding
`evaluation/big_tests/<project>/<variant>_<deconvolver>/` directory.

## Important comparison limitations

- The archived `tclean` commands are restart-stage commands (`calcres=False`,
  `calcpsf=False`). The CASA runner changes those to `True` and sets
  `restart=False` so an empty result directory can be reconstructed.
- When an archived pipeline `_targets.ms` is unavailable and the restored full
  `.ms` is used instead, the harness reads `CORRECTED_DATA`; the target MS would
  already have copied those calibrated values into its `DATA` column.
- CASA uses `auto-multithresh`; AstroViper runs both Hogbom and MT-MFS without
  that mask generator. Differences in the restored model therefore test both
  gridding/weighting and this known minor-cycle limitation.
- Full runs are expensive: the source datasets occupy roughly 7, 10, 53, and
  105 GB. Use `--niter 0` or a small value for an AstroViper smoke test before
  launching the archived iteration counts.
