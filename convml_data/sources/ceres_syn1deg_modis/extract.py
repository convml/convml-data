import warnings

import numpy as np
import rioxarray as rxr
from rasterio.errors import NotGeoreferencedWarning


def _extract_scene_dataset_from_hdf_file(fn_modis, timestamp):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        hdf_collection = rxr.open_rasterio(fn_modis)

    ds = hdf_collection[0]

    # make some more sensible names
    new_names = {
        v: ds[v]
        .long_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        for v in ds.data_vars
    }

    ds = ds.rename(new_names)

    # add lat/lon grid
    ds["lat"] = 90 - ds.y
    ds["lon"] = ds.x - 180
    ds = ds.swap_dims(dict(x="lon", y="lat"))

    # add time coord
    da = ds.observed_all_sky_toa_lw_flux
    ds["time"] = ("band",), [
        np.datetime64(da.RANGEBEGINNINGDATE) + np.timedelta64(n, "h")
        for n in range(int(da.band.count()))
    ]
    ds = ds.swap_dims(dict(band="time"))
    ds = ds.drop(["band", "spatial_ref"])

    # NB: scene timestamp might not be exactly the same as the time of this
    # dataset, but we're assuming here that we're selecting from a set of
    # timestamps that are close enough (that these scene has been associated
    # with this file previously)
    ds_scene = ds.sel(time=timestamp, method="nearest")

    return ds_scene


def extract_variable(task_input, var_name, timestamp):
    fn_modis = task_input.path

    ds_scene = _extract_scene_dataset_from_hdf_file(
        fn_modis=fn_modis, timestamp=timestamp
    )

    if var_name in ds_scene.data_vars:
        da = ds_scene[var_name]
    elif var_name == "toa_sw_cre":
        da = calc_toa_sw_cre(ds=ds_scene)
    elif var_name == "toa_lw_cre":
        da = calc_toa_lw_cre(ds=ds_scene)
    elif var_name == "toa_net_cre":
        da_toa_sw_cre = calc_toa_sw_cre(ds=ds_scene)
        da_toa_lw_cre = calc_toa_lw_cre(ds=ds_scene)
        da = da_toa_sw_cre + da_toa_lw_cre
        da.attrs["units"] = da_toa_lw_cre.units
        da.attrs["long_name"] = "TOA NET CRE"
    else:
        raise Exception(
            f"Variable `{var_name}` not found in MODIS SYN1Deg dataset. "
            f"Available variables are: {', '.join(sorted(ds_scene.data_vars))}"
        )

    da = da.sortby("lat")

    extra_attrs = ["INPUTPOINTER"]
    for attr in extra_attrs:
        if attr in da.attrs:
            del da.attrs[attr]

    return da


def calc_toa_sw_cre(ds):
    da_toa_sw_cre = (
        ds.observed_clear_sky_toa_albedo - ds.observed_all_sky_toa_albedo
    ) * ds.toa_sw_insolation
    da_toa_sw_cre.attrs["units"] = ds.toa_sw_insolation.units
    da_toa_sw_cre.attrs["long_name"] = "TOA SW CRE"
    da_toa_sw_cre.name = "toa_sw_cre"
    return da_toa_sw_cre


def calc_toa_lw_cre(ds):
    da_toa_lw_cre = -(
        ds.observed_all_sky_toa_lw_flux - ds.observed_clear_sky_toa_lw_flux
    )
    da_toa_lw_cre.attrs["units"] = ds.observed_clear_sky_toa_lw_flux.units
    da_toa_lw_cre.attrs["long_name"] = "TOA LW CRE"
    da_toa_lw_cre.name = "toa_lw_cre"
    return da_toa_lw_cre
