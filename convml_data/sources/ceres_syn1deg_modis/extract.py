import datetime
import difflib

import isodate
import pyhdf
import xarray as xr

"""
Parse date from SYN1Deg MODIS file.
The file has the following section (or something like it) inside its meta which we extract below
```
    OBJECT                 = RANGEBEGINNINGDATE
      NUM_VAL              = 1
      VALUE                = "2020-02-02"
    END_OBJECT             = RANGEBEGINNINGDATE

    OBJECT                 = RANGEENDINGDATE
      NUM_VAL              = 1
      VALUE                = "2020-02-02"
    END_OBJECT             = RANGEENDINGDATE
```
"""


def find_date(fh):
    lines = fh.attributes()["coremetadata"].splitlines()

    def _find_attr(attr):
        for i, line in enumerate(lines):
            if attr in line:
                k, v = [s.strip() for s in lines[i + 2].split("=")]
                assert k == "VALUE"
                return isodate.parse_date(v[1:-1])
                return k, v

    start_date = _find_attr("RANGEBEGINNINGDATE")
    end_date = _find_attr("RANGEENDINGDATE")
    assert start_date == end_date
    return start_date


def read_var_as_dataarray(fh, var_name, make_time_coord=True):
    dims, shape, _, _ = fh.datasets()[var_name]
    coords = {}
    if len(dims) > 1:
        for d in dims:
            coords[d] = read_var_as_dataarray(fh=fh, var_name=d)

    attrs = fh.select(var_name).attributes()
    values = fh.select(var_name).get()
    fill_value = attrs.pop("_FillValue", None)
    da = xr.DataArray(values, attrs=attrs, dims=dims, coords=coords)
    if fill_value is not None:
        da = da.where(da != fill_value)
    da.name = var_name

    if make_time_coord and "gmt_hr_index" in da.coords:
        date = find_date(fh=fh)
        t0 = datetime.datetime.combine(date, datetime.time(hour=0))
        da_hr = coords["gmt_hr_index"]
        values = [t0 + datetime.timedelta(hours=int(i)) for i in da_hr.values]
        da_time = xr.DataArray(values, dims=["gmt_hr_index"])
        da.coords["time"] = da_time
        da = da.swap_dims(dict(gmt_hr_index="time"))

    return da


def get_vars(fh):
    return list(fh.datasets())


def extract_variable(task_input, var_name, timestamp):
    if var_name == "toa_net_cre":
        da_toa_sw_cre = extract_variable(
            task_input=task_input, var_name="toa_sw_cre", timestamp=timestamp
        )
        da_toa_lw_cre = extract_variable(
            task_input=task_input, var_name="toa_lw_cre", timestamp=timestamp
        )
        da = da_toa_sw_cre + da_toa_lw_cre
        da.attrs["units"] = da_toa_lw_cre.units
        da.attrs["long_name"] = "TOA NET CRE"
        return da

    fn_modis = task_input.path
    fh = pyhdf.SD.SD(fn_modis)

    def _extract_var(v):
        da = read_var_as_dataarray(fh=fh, var_name=v, make_time_coord=True)
        da = da.rename(dict(latitude="lat", longitude="lon"))
        return da.sel(time=timestamp)

    all_var_names = get_vars(fh)

    if ":" in var_name:
        var_name, extra_args = var_name.split(":")
        k, v = extra_args.split("=")
        try:
            v = int(v)
        except ValueError:
            pass
        sel_kwargs = {k: v}
    else:
        sel_kwargs = None

    if var_name in all_var_names:
        da = _extract_var(v=var_name)
    elif var_name == "toa_sw_cre":
        da = calc_toa_sw_cre(
            da_observed_clear_sky_toa_albedo=_extract_var(v="obs_clr_toa_alb"),
            da_observed_all_sky_toa_albedo=_extract_var(v="obs_clr_toa_alb"),
            da_toa_sw_insolation=_extract_var(v="toa_sw_insol"),
        )
    elif var_name == "toa_lw_cre":
        da = calc_toa_lw_cre(
            da_observed_clear_sky_toa_lw_flux=_extract_var(v="obs_clr_toa_lw"),
            da_observed_all_sky_toa_lw_flux=_extract_var(v="obs_all_toa_lw"),
        )
    else:
        most_similar_vars = difflib.get_close_matches(
            var_name, all_var_names, n=10, cutoff=0.2
        )

        raise Exception(
            f"Variable `{var_name}` not found in MODIS SYN1Deg dataset. "
            f"Available variables are: {', '.join(all_var_names)}. "
            f"The most similar to the variable you requested are: {', '.join(most_similar_vars)}"
        )

    da = da.sortby("lat")

    if sel_kwargs is not None:
        da = da.sel(**sel_kwargs)

    extra_attrs = ["INPUTPOINTER"]
    for attr in extra_attrs:
        if attr in da.attrs:
            del da.attrs[attr]

    return da


def calc_toa_sw_cre(
    da_observed_clear_sky_toa_albedo,
    da_observed_all_sky_toa_albedo,
    da_toa_sw_insolation,
):
    da_toa_sw_cre = (
        da_observed_clear_sky_toa_albedo - da_observed_all_sky_toa_albedo
    ) * da_toa_sw_insolation
    da_toa_sw_cre.attrs["units"] = da_toa_sw_insolation.units
    da_toa_sw_cre.attrs["long_name"] = "TOA SW CRE"
    da_toa_sw_cre.name = "toa_sw_cre"
    return da_toa_sw_cre


def calc_toa_lw_cre(da_observed_all_sky_toa_lw_flux, da_observed_clear_sky_toa_lw_flux):
    da_toa_lw_cre = -(
        da_observed_all_sky_toa_lw_flux - da_observed_clear_sky_toa_lw_flux
    )
    da_toa_lw_cre.attrs["units"] = da_observed_clear_sky_toa_lw_flux.units
    da_toa_lw_cre.attrs["long_name"] = "TOA LW CRE"
    da_toa_lw_cre.name = "toa_lw_cre"
    return da_toa_lw_cre
