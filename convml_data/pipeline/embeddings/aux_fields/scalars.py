from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe
from eurec4a_environment import get_fields
from eurec4a_environment import nomenclature as nom
from eurec4a_environment.constants import cp_d, g
from eurec4a_environment.variables import atmos as atmos_variables
from eurec4a_environment.variables import tropical as tropical_variables
from eurec4a_environment.variables.utils import apply_by_column

from ...scene_sources import parse_scene_id
from . import datasources, utils

# from lagtraj.domain.sources.era5.utils import calculate_heights_and_pressures


levels_bl = dict(level=slice(120, None))
levels_cl = dict(level=slice(101, 120))


def calculate_heights_and_pressures(*args, **kwargs):
    raise NotImplementedError


def _get_regular_var(var_name, t, bbox_domain):
    crop_to_bl = False
    if var_name.endswith("_bl"):
        crop_to_bl = True
        var, _ = var_name.split("_")

    if var in ["q", "t"]:
        pass
    else:
        raise NotImplementedError(var)

    da = utils.clip_field(
        datasources.era5.get_var(t=t, var=var), bbox_domain=bbox_domain
    )

    if crop_to_bl:
        da = da.sel(**levels_bl)
    return da


def calc_lts(da_alt, da_p, da_theta):
    ds = xr.merge([da_alt, da_p, da_theta])

    return tropical_variables.lower_tropospheric_stability(
        ds=ds, vertical_coord="level"
    )


def calc_eis(da_alt, da_p, da_theta, da_d_theta__lts, da_t, da_z_lcl):
    ds = xr.merge([da_alt, da_p, da_theta, da_d_theta__lts, da_t, da_z_lcl])

    da_eis = tropical_variables.estimated_inversion_strength(
        ds=ds, vertical_coord="level", LTS=da_d_theta__lts.name, LCL=da_z_lcl.name
    )
    return da_eis


def find_LCL_Bolton(
    ds,
    temperature=nom.TEMPERATURE,
    rh=nom.RELATIVE_HUMIDITY,
    altitude=nom.ALTITUDE,
    vertical_coord=nom.ALTITUDE,
    layer_method="altitude_below_dtemp_min_height",
    debug=False,
    layer_sampling_method="half_minmax",
):
    """
    Calculates distribution of LCL from RH and T at different vertical levels
    returns mean LCL from this distribution
    Anna Lea Albright (refactored by Leif Denby) July 2020
    """

    def _calc_LCL_Bolton(ds_column):
        """
        Returns lifting condensation level (LCL) [m] calculated according to
        Bolton (1980).
        Inputs: da_temperature is temperature in Kelvin
                da_rh is relative humidity (0 - 1, dimensionless)
        Output: LCL in meters
        """
        # the calculation doesn't work where the relative humidity is zero, so
        # we drop those levels
        ds_column = ds_column.where(ds_column[rh] > 0.0, drop=True)
        da_temperature = ds_column[temperature]
        da_rh = ds_column[rh]
        da_altitude = ds_column[altitude]
        assert da_temperature.units == "K"
        assert da_rh.units == "1"
        assert da_altitude.units == "m"

        da_tlcl = (
            1.0 / ((1.0 / (da_temperature - 55.0)) - (np.log(da_rh) / 2840.0)) + 55.0
        )
        da_zlcl = da_altitude - (cp_d * (da_tlcl - da_temperature) / g)

        if layer_method == "altitude_below_dtemp_min_height":
            da_dtemp = da_temperature - da_tlcl
            z_dtemp_min = ds_column.isel({vertical_coord: da_dtemp.argmin().compute()})[
                altitude
            ]
            if debug:
                # da_dtemp["alt"] = ds_column[altitude]
                # da_dtemp.swap_dims(level="alt").sel(alt=slice(0, 4000)).plot(y="alt")
                # da_zlcl["alt"] = ds_column[altitude]
                # da_ = da_zlcl.where(da_altitude < z_dtemp_min, drop=True).compute().swap_dims(level="alt")
                # da_.plot(marker='.', y="alt", yincrease=True)
                pass
            # must be less-than-or-equal to include case where there is no
            # mixed-layer and surface state is the one that condenses at the
            # smallest height difference
            da_zlcl_layer = da_zlcl.where(da_altitude <= z_dtemp_min, drop=True)

        else:
            da_zlcl_layer = da_zlcl

        if layer_sampling_method == "mean":
            da_zlcl_column = np.mean(da_zlcl_layer)
        elif layer_sampling_method == "half_minmax":
            da_zlcl_column = 0.5 * (da_zlcl_layer.min() + da_zlcl_layer.max())
        else:
            raise NotImplementedError(layer_sampling_method)
        return da_zlcl_column

    ds_selected = get_fields(ds, **{temperature: "K", rh: "1", altitude: "m"})

    da_LCL = apply_by_column(
        ds=ds_selected, fn=_calc_LCL_Bolton, vertical_coord=vertical_coord
    )
    da_LCL.attrs["long_name"] = "lifting condensation level"
    da_LCL.attrs["reference"] = "Bolton 1980"
    da_LCL.attrs["units"] = "m"
    da_LCL.attrs["layer"] = layer_method
    da_LCL.attrs["layer_sampling"] = layer_sampling_method
    return da_LCL


def calc_lcl(da_alt, da_t, da_rh):
    ds = xr.merge([da_alt, da_t, da_rh])
    da_lcl = find_LCL_Bolton(
        ds=ds, altitude=da_alt.name, temperature=da_t.name, vertical_coord="level"
    )
    da_lcl.name = "z_lcl"
    return da_lcl


def _calc_eis(t, bbox_domain, var="eis"):
    ds = datasources.era5.get_dataset(
        t=t,
        variables=["q", "t", "z", "lnsp"],
        bbox_domain=bbox_domain,
    )

    ds["sp"] = np.exp(ds.lnsp)

    def add_height_and_pressure(ds):
        ds_aux = calculate_heights_and_pressures(ds=ds)
        ds["alt"] = ds_aux["height_f"]
        ds["p"] = ds_aux["p_f"]
        return ds

    add_height_and_pressure(ds)

    ds[nom.POTENTIAL_TEMPERATURE] = atmos_variables.potential_temperature(
        ds=ds, vertical_coord="level"
    )

    da_lts = tropical_variables.lower_tropospheric_stability(
        ds=ds, vertical_coord="level"
    )
    ds["dtheta_LTS"] = da_lts

    if var == "lts":
        return da_lts

    ds["z_LCL"] = 700.0 * xr.ones_like(ds.t.isel(level=0), dtype=float)
    ds["z_LCL"].attrs["units"] = "m"

    da_eis = tropical_variables.estimated_inversion_strength(
        ds=ds, vertical_coord="level"
    )
    return da_eis


def get_var_more(t, bbox_domain, var):
    ds = datasources.era5.get_dataset(
        t=t,
        variables=["q", "t", "z", "lnsp"],
        bbox_domain=bbox_domain,
    )

    ds["sp"] = np.exp(ds.lnsp)

    def add_height_and_pressure(ds):
        ds_aux = calculate_heights_and_pressures(ds=ds)
        ds["alt"] = ds_aux["height_f"]
        ds["p"] = ds_aux["p_f"]
        return ds

    add_height_and_pressure(ds)
    return ds[[var, "alt"]]


def main(da_emb, scene_id, var_name, deg_pad=1.0):
    t = parse_scene_id(scene_id)

    bbox_domain = dict(
        lon=slice(da_emb.lon.min() - deg_pad, da_emb.lon.max() + deg_pad),
        lat=slice(da_emb.lat.min() - deg_pad, da_emb.lat.max() + deg_pad),
    )

    file_id = f"{var_name}_era5_rect_{scene_id}"

    fn_crop = Path(f"{file_id}__crop.nc")

    if not fn_crop.exists():
        if var_name in ["q_bl", "t_bl"]:
            da = _get_regular_var(var_name=var_name, t=t, bbox_domain=bbox_domain)
        elif var_name in ["q", "t"]:
            da = get_var_more(t=t, bbox_domain=bbox_domain, var=var_name)
        elif var_name in ["lts", "eis"]:
            da = _calc_eis(t=t, bbox_domain=bbox_domain, var=var_name).transpose(
                "lat", "lon"
            )
        else:
            raise NotImplementedError(var_name)

        da.to_netcdf(fn_crop)
        da.close()

    da = xr.open_dataset(str(fn_crop))

    regridder = xe.Regridder(
        da,
        da_emb,
        "bilinear",
    )

    da_rect = regridder(da)

    da_rect["x"] = da_emb.x
    da_rect["y"] = da_emb.y
    da_rect.to_netcdf(f"{file_id}.nc")

    da.close()

    # fig, ax = plt.subplots(ncols=1, figsize=(8,8))

    # da_rect.plot(ax=ax)
    # plt.savefig(f"{file_id}.png")
    print(f"processed {file_id}")

    # In[48]:


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("fn_emb")
    argparser.add_argument("scene_id")
    argparser.add_argument("var_name")
    args = argparser.parse_args()

    da_emb = xr.open_dataarray(args.fn_emb)
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        main(da_emb=da_emb, scene_id=args.scene_id, var_name=args.var_name)
