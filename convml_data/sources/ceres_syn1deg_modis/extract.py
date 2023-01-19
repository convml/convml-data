import warnings

import numpy as np
import rioxarray as rxr
from rasterio.errors import NotGeoreferencedWarning


def extract_variable(task_input, var_name):
    fn_modis = task_input.path

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

    if var_name in ds.data_vars:
        da = ds[var_name]
    elif var_name == "toa_sw_cre":
        da = calc_toa_sw_cre(ds=ds)
    elif var_name == "toa_lw_cre":
        da = calc_toa_lw_cre(ds=ds)
    elif var_name == "toa_net_cre":
        da_toa_sw_cre = calc_toa_sw_cre(ds=ds)
        da_toa_lw_cre = calc_toa_lw_cre(ds=ds)
        da = da_toa_sw_cre + da_toa_lw_cre
        da.attrs["units"] = da_toa_lw_cre.units
        da.attrs["long_name"] = "TOA NET CRE"
    else:
        raise Exception(
            f"Variable `{var_name}` not found in MODIS SYN1Deg dataset. "
            f"Available variables are: {', '.join(ds.data_vars)}"
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
