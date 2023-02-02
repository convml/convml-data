# import numpy as np
import datetime

from .sw_flux import calc_reflected_sw_from_albedo, calc_toa_total_outgoing_flux


def compute_derived_variable(ds, var_name):  # fp_scene, fp_dest, var_name):
    time = ds.time.values.astype("datetime64[s]").astype(datetime.datetime)

    if var_name == "approximate_toa_total_outgoing_radiation_flux":
        da_sw_albedo = ds.broadband_shortwave_albedo
        da_lw_flux = ds.broadband_longwave_flux
        da = calc_toa_total_outgoing_flux(
            da_sw_albedo=da_sw_albedo, da_lw_flux=da_lw_flux, dt_utc=time
        )
    elif var_name == "broadband_shortwave_flux":
        da_sw_albedo = ds.broadband_shortwave_albedo
        da = calc_reflected_sw_from_albedo(da_sw_albedo=da_sw_albedo, dt_utc=time)
        da.name = "broadband_shortwave_flux"
    else:
        raise NotImplementedError(var_name)

    return da
