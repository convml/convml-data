import datetime

import numpy as np
import xarray as xr


def npdt64_to_dt(v):
    return v.astype("datetime64[s]").astype(datetime.datetime)


def find_nearest_time(t, times, return_index=False):
    """
    Find nearest time in `times` to target time `t`, both are expected to of
    type `np.datetime64`
    """
    if not isinstance(times, xr.DataArray):
        da_times = xr.DataArray(times, dims=("time"))
        t_dim = "time"
    else:
        da_times = times
        t_dim = times.dims[0]

    if not isinstance(t, xr.DataArray):
        t = xr.DataArray(t)

    da_delta_t = np.abs(da_times - t)
    idx_t = da_delta_t.argmin()
    da_t_nearest = da_times.isel({t_dim: idx_t})

    t_nearest = npdt64_to_dt(da_t_nearest.values)
    if return_index:
        return t_nearest, idx_t.astype(int).item()

    return t_nearest
