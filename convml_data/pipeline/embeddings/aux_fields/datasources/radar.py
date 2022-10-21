import pickle

import xarray as xr


def read_with_crs(fn):
    crs = pickle.load(open(f"{fn}.crs", "rb"))
    da = xr.open_dataarray(fn)
    da.attrs["crs"] = crs
    return da
