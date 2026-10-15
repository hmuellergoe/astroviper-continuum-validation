# Corrected TW Hydra beam reference

The original frozen cube remains unchanged. Its beam metadata predates AstroVIPER commit `b2cf314a341aeaf824a87b7d51b7880a18e83d10` (25 August 2026). That commit fixes the physical scale of the resampled fitting grid: N pixels span N−1 intervals.

`twhya_corrected_beam.npy` was generated on 20 September 2026 by refitting the **original frozen PSF**, not the current imaging output, with the corrected fitter at revision `795b8ba`. The historical pre-fix fitter reproduces the original beam metadata to 2.55e-10 maximum relative difference. The corrected fitter reproduces the 18 September output to 2.71e-10. The historical/new width difference reaches 4.14%.

Reproduction: `results/diagnostics_2026-09-20/diagnose_beam.py`; log: `beam.log`. The dirty-cube test retains its original 1e-6 tolerance and original pixel/coordinate reference, but compares beam metadata with this independently refitted reference.

## Primary-beam image reference

The original frozen cube's `PRIMARY_BEAM` predates AstroVIPER commit
`40d773de57437cf914367152f2c6b6f9b506acd3` (20 August 2026). That intentional
change aligned the image convention with CASA by replacing the voltage pattern
`V` with the power sensitivity pattern `P = |V|^2`; it also adopted the
CASA-compatible Airy response.

`twhya_power_primary_beam.npz` freezes the five-channel power-pattern output
from the canonical TW Hydra geometry at revision `1f5807e`. The AstroVIPER
primary-beam unit tests independently verify the analytic response against the
CASA prescription. Because the beam is polarization independent, the compact
reference stores one polarization plane and the regression compares it against
both output polarizations at the original `1e-6` tolerance. The original cube
store remains unchanged so that residual and PSF history are preserved.
