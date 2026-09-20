# Corrected TW Hydra beam reference

The original frozen cube remains unchanged. Its beam metadata predates AstroVIPER commit `b2cf314a341aeaf824a87b7d51b7880a18e83d10` (25 August 2026). That commit fixes the physical scale of the resampled fitting grid: N pixels span N−1 intervals.

`twhya_corrected_beam.npy` was generated on 20 September 2026 by refitting the **original frozen PSF**, not the current imaging output, with the corrected fitter at revision `795b8ba`. The historical pre-fix fitter reproduces the original beam metadata to 2.55e-10 maximum relative difference. The corrected fitter reproduces the 18 September output to 2.71e-10. The historical/new width difference reaches 4.14%.

Reproduction: `results/diagnostics_2026-09-20/diagnose_beam.py`; log: `beam.log`. The dirty-cube test retains its original 1e-6 tolerance and original pixel/coordinate reference, but compares beam metadata with this independently refitted reference.
