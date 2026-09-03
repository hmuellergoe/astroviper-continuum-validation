# Evaluation programs

The top-level Python programs evaluate continuum-demonstrator results placed
under `results/<project>/<variant>/`. Large Measurement Sets and generated
image stores are intentionally excluded from this repository.

- `evaluate_science_metrics.py` measures beam area, image noise, recovered
  flux, spectral index, dynamic range, data coverage, and applicable ALMA QA2
  criteria.
- `render_continuum_images.py` renders restored images, center zooms, and
  diagnostic panels for completed AstroVIPER configurations.
- `compare_casa_astroviper.py` compares exact-input CASA benchmark products
  with selected AstroVIPER MFS and MVC products.
- `compare_casa_algorithm_matched.py` evaluates the retained algorithm-matched
  CASA cases, including the GS8 masking variants.
- `test_evaluate_science_metrics.py` checks the QA2 boundaries and result
  discovery behavior without requiring the large datasets.

Run the science evaluation and image renderer from the repository root:

```bash
$VALIDATION_PYTHON evaluation/evaluate_science_metrics.py --all
$VALIDATION_PYTHON evaluation/render_continuum_images.py
```

The evaluators write generated reports and plots below
`evaluation/<project>/<variant>/`. Existing harness-specific evaluation code
remains organized under `light_tests/`, `medium_tests/`, and `big_tests/`.
