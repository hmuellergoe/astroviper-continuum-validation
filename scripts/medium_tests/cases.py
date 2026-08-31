"""Shared CASA/RADPS continuum validation case definitions."""

from __future__ import annotations

CASES = {
    "phase_center": {
        "ms": "refim_point.ms",
        "field": "0",
        "field_name": "fake",
        "imsize": 200,
        "cell_arcsec": 10.0,
        "phasecenter": None,
        "reffreq_hz": 1.5e9,
        "nchan_mvc": 20,
        "pblimit": 0.1,
        "description": (
            "Twenty-channel phase-center point source with spectral index -1; "
            "adapted from CASA test_mtmfsviacube_compare_with_mfs."
        ),
    },
    "off_axis": {
        "ms": "refim_oneshiftpoint.mosaic.ms",
        "field": "0",
        "field_name": "fake0",
        "imsize": 512,
        "cell_arcsec": 10.0,
        "phasecenter": "J2000 19h59m28.523 +40d54m01.152",
        "reffreq_hz": 1.5e9,
        "nchan_mvc": 3,
        "pblimit": 0.1,
        "description": (
            "Single selected field containing an off-axis point source; uses "
            "the standard gridder to exercise the MVC primary-beam convention."
        ),
    },
}

WEIGHTING_CASES = {
    "natural_local": {
        "weighting": "natural",
        "weighting_scope": "local",
        "casa_weighting": "natural",
        "robust": None,
    },
    "briggs_local": {
        "weighting": "briggs",
        "weighting_scope": "local",
        "casa_weighting": "briggs",
        "robust": 0.5,
    },
    "briggs_global": {
        "weighting": "briggs",
        "weighting_scope": "global",
        "casa_weighting": "briggs",
        "robust": 0.5,
    },
}

CLEANING_CASE = {
    "threshold_jy": 0.1,
    "niter": 100,
    "nmajor": 5,
    "description": (
        "Clean the 1.06 Jy phase-center point source to 0.1 Jy so both CASA "
        "and AstroVIPER must produce a nonzero model."
    ),
}


def selected_cases(names):
    """Return validated case definitions in the requested order."""
    unknown = set(names) - set(CASES)
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [(name, CASES[name]) for name in names]


def selected_weightings(names):
    """Return validated weighting definitions in the requested order."""
    unknown = set(names) - set(WEIGHTING_CASES)
    if unknown:
        raise ValueError(f"Unknown weighting cases: {sorted(unknown)}")
    return [(name, WEIGHTING_CASES[name]) for name in names]


def weighting_suffix(name):
    """Keep existing natural-result names while naming added variants."""
    return "" if name == "natural_local" else f"_{name}"


def cleaning_suffix(clean):
    """Keep dirty-reference names stable and isolate threshold-cleaned outputs."""
    return "_clean" if clean else ""
