import numpy as np
import xarray as xr
from metpy import calc as mpcalc

from .scalars import calc_eis, calc_lcl, calc_lts

levels_bl = dict(level=slice(120, None))
levels_cl = dict(level=slice(101, 120))

SURFACE_VARIABLES = dict(
    sst="sea surface temperature",
    lnsp="natural logarithm of the surface pressure",
    z="surface altitude",
)
MODEL_LEVEL_VARIABLES = dict(
    u="zonal wind",
    v="meridional wind",
    t="potential temperature",
    q="total moisture",
    rh="relative humidity",
)
SOURCE_VARIABLES = dict(**SURFACE_VARIABLES, **MODEL_LEVEL_VARIABLES)


def _calc_umag(da_u, da_v):
    da = np.sqrt(da_v**2.0 + da_u**2.0)
    da.name = "umag"
    da.attrs["long_name"] = "windspeed"
    da.attrs["units"] = da_u.units
    return da


def _calc_rh(da_p, da_t, da_q):
    rh_arr = mpcalc.relative_humidity_from_specific_humidity(
        pressure=da_p, temperature=da_t, specific_humidity=da_q
    )
    assert rh_arr.units == 1
    da_rh = xr.DataArray(
        np.array(rh_arr),
        coords=da_t.coords,
        dims=da_t.dims,
        attrs=dict(
            long_name="relative humidity", units="1", standard_name="relative_humidity"
        ),
    )
    da_rh.name = "rh"
    return da_rh


DERIVED_VARIABLES = dict(
    d_theta__lts=(calc_lts, ["alt", "p", "theta"]),
    d_theta__eis=(calc_eis, ["alt", "p", "theta", "d_theta__lts", "t", "z_lcl"]),
    umag=(_calc_umag, ["u", "v"]),
    rh=(_calc_rh, ["p", "t", "q"]),
    z_lcl=(calc_lcl, ["alt", "t", "rh"]),
)
