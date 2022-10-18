import numpy as np
import xarray as xr
from metpy import calc as mpcalc

from .scalars import calc_eis, calc_lcl, calc_lts
from .utils import calculate_heights_and_pressures

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
)
SOURCE_VARIABLES = dict(**SURFACE_VARIABLES, **MODEL_LEVEL_VARIABLES)


def _calc_umag(da_u, da_v):
    da = np.sqrt(da_v**2.0 + da_u**2.0)
    da.name = "umag"
    da.attrs["long_name"] = "windspeed"
    da.attrs["units"] = da_u.units
    return da


def _calc_alt_p(da_t, da_q, da_z, da_lnsp):
    """
    Calculate heights and pressures at half-levels for ERA5 files
    """
    da_sp = np.exp(da_lnsp)
    da_sp.name = "sp"

    ds = xr.Dataset(dict(t=da_t, q=da_q, z=da_z, sp=da_sp))
    ds_aux = calculate_heights_and_pressures(ds=ds)

    ds = ds_aux[["height_f", "p_f"]].rename(dict(height_f="alt", p_f="p"))
    return ds


def _calc_rh(da_t, da_q, ds_alt_p):
    da_p = ds_alt_p.p
    da_rh = mpcalc.relative_humidity_from_specific_humidity(
        pressure=da_p, temperature=da_t, specific_humidity=da_q
    )
    assert isinstance(da_rh, xr.DataArray)
    da_rh.attrs["long_name"] = "relative humidity"
    da_rh.attrs["standard_name"] = "relative_humidity"
    da_rh.name = "rh"
    return da_rh


DERIVED_VARIABLES = dict(
    d_theta__lts=(calc_lts, ["alt", "p", "theta"]),
    d_theta__eis=(calc_eis, ["alt", "p", "theta", "d_theta__lts", "t", "z_lcl"]),
    umag=(_calc_umag, ["u", "v"]),
    rh=(_calc_rh, ["t", "q", "alt_p"]),
    z_lcl=(calc_lcl, ["alt", "t", "rh"]),
    alt_p=(_calc_alt_p, ["q", "t", "z", "lnsp"]),
)
