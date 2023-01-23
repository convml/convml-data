from functools import partial

import numpy as np
import xarray as xr
from eurec4a_environment.variables import atmos as atmos_variables
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


def _calc_rh(da_t, da_q, da_p):
    da_rh = mpcalc.relative_humidity_from_specific_humidity(
        pressure=da_p, temperature=da_t, specific_humidity=da_q
    )
    assert isinstance(da_rh, xr.DataArray)
    da_rh.attrs["long_name"] = "relative humidity"
    da_rh.attrs["standard_name"] = "relative_humidity"
    assert da_rh.metpy.units == "1"
    da_rh.attrs["units"] = "1"
    da_rh.name = "rh"
    return da_rh


def _calc_potential_temperature(da_p, da_t):
    ds = xr.merge([da_p, da_t])
    da_theta = atmos_variables.potential_temperature(ds=ds, vertical_coord="level")
    return da_theta


def _calc_alt(ds_alt_p):
    return ds_alt_p.alt


def _calc_pressure(ds_alt_p):
    return ds_alt_p.p


def _calc_layer_umag(da_u, da_v, layer_name):
    ds = xr.merge([da_u, da_v])
    if layer_name == "boundary_layer":
        kwargs = levels_bl
    elif layer_name == "cloud_layer":
        kwargs = levels_cl
    else:
        raise NotImplementedError(layer_name)

    ds_layer = ds.sel(**kwargs)
    da_layer_umag = _calc_umag(da_u=ds_layer.u, da_v=ds_layer.v).mean(
        dim="level", keep_attrs=True
    )
    da_layer_umag.attrs[
        "long_name"
    ] = f"{layer_name} ({kwargs['level'].start} - {kwargs['level'].stop}) mean windspeed"
    return da_layer_umag


def _calc_layer_qmean(da_q, layer_name):
    if layer_name == "boundary_layer":
        kwargs = levels_bl
    elif layer_name == "cloud_layer":
        kwargs = levels_cl
    else:
        raise NotImplementedError(layer_name)

    da_layer_q = da_q.sel(**kwargs).mean(dim="level", keep_attrs=True)
    da_layer_q.attrs[
        "long_name"
    ] = f"{layer_name} ({kwargs['level'].start} - {kwargs['level'].stop}) mean total moisture"
    return da_layer_q


def _calc_bl_umag(da_u, da_v):
    return _calc_layer_umag(da_u=da_u, da_v=da_v, layer_name="boundary_layer")


def _calc_cl_umag(da_u, da_v):
    return _calc_layer_umag(da_u=da_u, da_v=da_v, layer_name="cloud_layer")


def _calc_layer_wind_direction(da_u, da_v, layer_name):
    ds = xr.merge([da_u, da_v])
    if layer_name == "boundary_layer":
        kwargs = levels_bl
    elif layer_name == "cloud_layer":
        kwargs = levels_cl
    else:
        raise NotImplementedError(layer_name)

    ds_layer = ds.sel(**kwargs)
    ds_layer_mean = ds_layer.mean(dim="level", keep_attrs=True)
    da_layer_mean_wind_direction = mpcalc.wind_direction(
        u=ds_layer_mean.u, v=ds_layer_mean.v
    )
    da_layer_mean_wind_direction.attrs[
        "long_name"
    ] = f"{layer_name} ({kwargs['level'].start} - {kwargs['level'].stop}) mean wind direction"
    return da_layer_mean_wind_direction


def _calc_bl_wind_direction(da_u, da_v):
    return _calc_layer_wind_direction(da_u=da_u, da_v=da_v, layer_name="boundary_layer")


def _calc_cl_wind_direction(da_u, da_v):
    return _calc_layer_wind_direction(da_u=da_u, da_v=da_v, layer_name="cloud_layer")


def _calc_column_total_precipitable_water(da_q, da_p):
    da_dp = da_p.differentiate(coord="level")
    da_tpw = 1.0 / 9.8 * (da_q * da_dp).sum(dim="level")
    da_tpw.attrs["long_name"] = "total precipitable water"
    da_tpw.attrs["units"] = "mm"
    return da_tpw


DERIVED_VARIABLES = dict(
    umag=(_calc_umag, ["u", "v"]),
    bl_wind_direction=(_calc_bl_wind_direction, ["u", "v"]),
    cl_wind_direction=(_calc_cl_wind_direction, ["u", "v"]),
    alt_p=(_calc_alt_p, ["q", "t", "z", "lnsp"]),
    alt=(_calc_alt, ["alt_p"]),
    p=(_calc_pressure, ["alt_p"]),
    rh=(_calc_rh, ["t", "q", "p"]),
    z_lcl=(calc_lcl, ["alt", "t", "rh"]),
    theta=(_calc_potential_temperature, ["p", "t"]),
    d_theta__lts=(calc_lts, ["alt_p", "theta"]),
    d_theta__eis=(calc_eis, ["alt_p", "theta", "d_theta__lts", "t", "z_lcl"]),
    bl_umag=(
        partial(_calc_layer_umag, layer_name="boundary_layer"),
        ["u", "v"],
    ),
    cl_umag=(partial(_calc_layer_umag, layer_name="cloud_layer"), ["u", "v"]),
    bl_qmean=(
        partial(_calc_layer_qmean, layer_name="boundary_layer"),
        ["q"],
    ),
    cl_qmean=(
        partial(_calc_layer_qmean, layer_name="cloud_layer"),
        ["q"],
    ),
    tpw=(_calc_column_total_precipitable_water, ["q", "p"]),
)
