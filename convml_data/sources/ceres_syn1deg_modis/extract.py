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
        v: ds[v].long_name.lower().replace(" ", "_").replace("-", "_")
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
    else:
        raise Exception(
            f"Variable `{var_name}` not found in MODIS SYN1Deg dataset. "
            f"Available variables are: {', '.join(ds.data_vars)}"
        )

    da = da.sortby("lat")

    extra_attrs = ["INPUTPOINTER"]
    for attr in extra_attrs:
        del da.attrs[attr]

    return da
