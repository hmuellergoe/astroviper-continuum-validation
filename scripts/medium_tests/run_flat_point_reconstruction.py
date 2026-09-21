"""Matched nterms=1, natural-weight MFS/Hogbom example; run engines separately."""

from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGES = {"dirty": 0, "update": 5, "clean": 100}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["casa", "astroviper"], required=True)
    p.add_argument(
        "--ms", type=Path, default=ROOT / "data/medium_tests/refim_flat_point.ms"
    )
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/medium_tests/flat_point"
    )
    a = p.parse_args()
    a.ms = a.ms.resolve()
    a.output_dir = a.output_dir.resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for stage, niter in STAGES.items():
        prefix = a.output_dir / f"{a.engine}_{stage}"
        if any(a.output_dir.glob(prefix.name + ".*")):
            raise FileExistsError(
                f"{prefix}: use a fresh output directory to avoid CASA restart state"
            )
        if a.engine == "casa":
            from casatasks import tclean
            from casatools import image

            result = tclean(
                vis=str(a.ms),
                imagename=str(prefix),
                field="0",
                datacolumn="data",
                imsize=200,
                cell="10arcsec",
                stokes="I",
                specmode="mfs",
                reffreq="1.5GHz",
                gridder="standard",
                deconvolver="hogbom",
                nterms=1,
                weighting="natural",
                niter=niter,
                cycleniter=5,
                nmajor=20,
                threshold="0.01Jy",
                gain=0.1,
                cyclefactor=1.0,
                minpsffraction=0.05,
                maxpsffraction=0.8,
                mask="circle[[100pix,100pix],3pix]",
                restoration=True,
                pbcor=True,
                pblimit=0.1,
                parallel=False,
            )
            products = {}
            for name, suffix in [
                ("model", "model"),
                ("residual", "residual"),
                ("psf", "psf"),
                ("pb", "pb"),
                ("restored", "image"),
                ("pbcor", "image.pbcor"),
            ]:
                t = image()
                t.open(str(prefix) + "." + suffix)
                products[name] = t.getchunk().squeeze()
                if name == "restored":
                    products["beam_json"] = np.asarray(json.dumps(t.restoringbeam()))
                t.close()
            summary = {
                k: str(v)
                for k, v in result.items()
                if k in ["iterdone", "nmajordone", "stopcode", "summaryminor"]
            }
        else:
            from run_radps_reference import (
                convert_ms,
                phase_direction_from_ps,
                require_one_processing_set_field,
            )
            from astroviper.distributed_applications.imaging import (
                image_continuum_single_field,
            )
            import xarray as xr

            ps = a.output_dir / "flat.ps.zarr"
            convert_ms(a.ms, ps, False)
            field = require_one_processing_set_field(ps)
            direction, freq = phase_direction_from_ps(ps)
            xx, yy = np.indices((200, 200))
            mask = (xx - 100) ** 2 + (yy - 100) ** 2 <= 9
            np.save(a.output_dir / "clean_mask.npy", mask)
            result = image_continuum_single_field(
                ps_store=str(ps),
                image_store=str(prefix) + ".img.zarr",
                image_params=dict(
                    image_size=[200, 200],
                    cell_size=np.array([-10.0, 10.0]) * np.pi / (180 * 3600),
                    phase_direction=direction,
                    frequency_coords=freq,
                    polarization_coords=["I"],
                    time_coords=[0],
                    spectral_reference="topo",
                    fft_padding=1.2,
                    cpp_gridder=True,
                    reference_frequency_hz=1.5e9,
                    reference_frequency=1.5e9,
                    nterms=1,
                ),
                imaging_weights_params=dict(
                    weighting="natural",
                    weighting_scope="local",
                    casa_weighting_implementation=True,
                ),
                iteration_control_params=dict(
                    niter=max(1, niter),
                    nmajor=20 if niter else 0,
                    threshold=0.01 if niter else 1.0e30,
                    gain=0.1,
                    cyclefactor=1.0,
                    cycleniter=5,
                    minpsffraction=0.05,
                    maxpsffraction=0.8,
                ),
                gridder="prolate_spheroidal",
                deconvolver="hogbom",
                restore=True,
                pbcor=True,
                pblimit=0.1,
                clean_mask=str(a.output_dir / "clean_mask.npy"),
                specmode="mfs",
                field_name=field,
                scan_intents=["None"],
                image_data_variables_keep=[
                    "sky_model",
                    "sky_residual",
                    "point_spread_function",
                    "primary_beam",
                    "beam_fit_params_point_spread_function",
                    "sky_restored",
                ],
                processing_set_data_group_name="base",
                single_precision_image=False,
                processing_function_threads=1,
                n_chunks=2,
                overwrite=True,
                memory_mode="in_memory",
                visibility_memory_mode="recompute",
                vizualize_graph=False,
                compute_backend="dask",
                reduce_mode="tree",
                reduce_n_batch=2,
            )
            ds = xr.open_zarr(str(prefix) + ".img.zarr")
            print("PRODUCTS", list(ds.data_vars), flush=True)
            names = dict(
                model="SKY_MODEL",
                residual="SKY_RESIDUAL",
                psf="POINT_SPREAD_FUNCTION",
                pb="PRIMARY_BEAM",
                restored="SKY_RESTORED",
                pbcor="SKY_RESTORED_PBCOR",
            )
            products = {
                k: ds[v].isel(polarization=0).values.squeeze() for k, v in names.items()
            }
            products["beam"] = ds.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.isel(
                polarization=0
            ).values.squeeze()
            summary = {
                k: str(result[k])
                for k in ["n_major_cycles", "stopcode", "deconvolution"]
                if k in result
            }
        np.savez_compressed(str(prefix) + ".npz", **products)
        Path(str(prefix) + ".json").write_text(json.dumps(summary, indent=2))
        print("COMPLETED", a.engine, stage, summary, flush=True)


if __name__ == "__main__":
    main()
