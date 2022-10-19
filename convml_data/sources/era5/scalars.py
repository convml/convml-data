import numpy as np
import scipy.signal
import xarray as xr
from eurec4a_environment import get_fields
from eurec4a_environment import nomenclature as nom
from eurec4a_environment.constants import cp_d, g
from eurec4a_environment.variables import atmos as atmos_variables
from eurec4a_environment.variables import tropical as tropical_variables
from eurec4a_environment.variables.utils import apply_by_column

levels_bl = dict(level=slice(120, None))
levels_cl = dict(level=slice(101, 120))


def calc_lts(ds_alt_p, da_theta):
    ds = xr.merge([ds_alt_p, da_theta]).squeeze()
    return tropical_variables.lower_tropospheric_stability(
        ds=ds, vertical_coord="level"
    )


def calc_eis(ds_alt_p, da_theta, da_d_theta__lts, da_t, da_z_lcl):
    ds = xr.merge([ds_alt_p, da_theta, da_d_theta__lts, da_t, da_z_lcl])

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
    layer_method="first_delta_temp_minimum",
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
        da_delta_temp = da_tlcl - da_temperature
        da_zlcl = da_altitude - (cp_d * da_delta_temp / g)

        if layer_method == "first_delta_temp_minimum":
            # find the altitude at which the there is a local minimum in the difference
            # between the temperature and LCL temperature
            da_dtemp = da_temperature - da_tlcl
            alt_peaks = da_altitude.isel(
                {vertical_coord: scipy.signal.find_peaks(da_delta_temp)[0]}
            )
            alt_min_alt_peak = alt_peaks.min()
            # must be less-than-or-equal to include case where there is no
            # mixed-layer and surface state is the one that condenses at the
            # smallest height difference
            da_zlcl_layer = da_zlcl.where(da_altitude <= alt_min_alt_peak, drop=True)
        elif layer_method == "altitude_below_dtemp_min_height":
            # NOTE: this method can sometimes produce a very deep layer since there
            # is sometimes a minimum at high altitude
            da_dtemp = da_temperature - da_tlcl
            vert_idx = da_dtemp.argmin().compute()
            z_dtemp_min = ds_column.isel({vertical_coord: vert_idx})[altitude]
            # must be less-than-or-equal to include case where there is no
            # mixed-layer and surface state is the one that condenses at the
            # smallest height difference
            da_zlcl_layer = da_zlcl.where(da_altitude <= z_dtemp_min, drop=True)
        elif layer_method is None:
            da_zlcl_layer = da_zlcl
        else:
            raise NotImplementedError(layer_method)

        if layer_sampling_method == "mean":
            da_zlcl_column = np.mean(da_zlcl_layer)
        elif layer_sampling_method == "median":
            da_zlcl_column = np.median(da_zlcl_layer)
        elif layer_sampling_method == "half_minmax":
            da_zlcl_column = 0.5 * (da_zlcl_layer.min() + da_zlcl_layer.max())
        else:
            raise NotImplementedError(layer_sampling_method)

        da_zlcl_column.attrs["layer_std"] = da_zlcl_layer.std().item()
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


def _calc_eis(da_q, da_t, da_z, da_lnsp, da_z_lcl):
    ds = xr.merge([da_q, da_t, da_z, da_lnsp])

    ds[nom.POTENTIAL_TEMPERATURE] = atmos_variables.potential_temperature(
        ds=ds, vertical_coord="level"
    )

    da_lts = tropical_variables.lower_tropospheric_stability(
        ds=ds, vertical_coord="level"
    )
    ds["dtheta_LTS"] = da_lts
    ds["z_LCL"] = da_z_lcl

    da_eis = tropical_variables.estimated_inversion_strength(
        ds=ds, vertical_coord="level"
    )
    return da_eis
