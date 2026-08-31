# Lightweight continuum tests

These tests use small synthetic xarray datasets and mirror AstroVIPER's layer
structure. From an AstroVIPER checkout containing the continuum implementation,
run them with:

```bash
pytest /path/to/astroviper-continuum-validation/scripts/light_tests
```

They cover continuum frequency reduction, cached node state, reference-frequency
primary beams, MVC normal equations, and continuum imaging-weight grid indexing.
