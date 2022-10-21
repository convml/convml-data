import datetime
from pathlib import Path

import luigi
import numpy as np
import xarray as xr
from eurec4a_environment.variables import atmos as atmos_variables

# from lagtraj.domain.sources.era5.utils import calculate_heights_and_pressures
from metpy import calc as mpcalc

from .. import utils
from ..scalars import calc_eis, calc_lcl, calc_lts

levels_bl = dict(level=slice(120, None))
levels_cl = dict(level=slice(101, 120))

SURFACE_VARIABLES = ["sst"]


def calculate_heights_and_pressures(*args, **kwargs):
    raise NotImplementedError


def _make_src_filename(
    t=datetime.datetime(year=2020, month=1, day=30, hour=4, minute=0),
    var="t",
):
    if var in SURFACE_VARIABLES:
        level = "sfc"
    else:
        level = "ml"
    fn = t.strftime(
        "/badc/ecmwf-era5/data/oper/an_{level}/%Y/%m/%d/ecmwf-era5_oper_an_{level}_%Y%m%d%H%M.{var}.nc"
    ).format(var=var, level=level)
    return fn


class ERA5Field(luigi.Task):
    scene_id = luigi.Parameter()
    var_name = luigi.Parameter()

    def output(self):
        if self.var_name not in "u v q t lnsp z sst".split(" "):
            raise NotImplementedError(self.var_name)

        t = utils.parse_scene_id(self.scene_id)

        # era5 is available every hour so we truncate to nearest hour
        add_hour = False
        if t.minute > 30:
            add_hour = True

        t_hr = datetime.datetime(
            year=t.year, month=t.month, day=t.day, hour=t.hour, minute=0
        )

        if add_hour:
            t_hr += datetime.timedelta(hour=1)

        fn = _make_src_filename(t=t_hr, var=self.var_name)
        return utils.XArrayTarget(fn)


def crop_field(da, bbox_domain):
    lat = bbox_domain["lat"]
    # era5 data is sorted so that lats are decreasing in value
    if lat[0] < lat[1]:
        lat = [lat[1], lat[0]]

    lat = slice(*lat)
    lon = slice(*bbox_domain["lon"])

    da_ = da.sel(lat=lat).roll(dict(lon=len(da.lon) // 2), roll_coords=True)
    da_ = da_.assign_coords(lon=da_.lon.where(da_.lon < 180.0, da_.lon - 360.0))
    da_ = da_.sel(lon=lon)
    return da_


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


class ERA5FieldCropped(luigi.Task):
    scene_id = luigi.Parameter()
    var_name = luigi.Parameter()
    bbox = utils.BBoxParameter()
    datapath = luigi.Parameter()

    def requires(self):
        if self.var_name in ["alt", "p"]:
            return ERA5HeightAndPressureFieldCropped(
                scene_id=self.scene_id,
                bbox=self.bbox,
                datapath=self.datapath,
            )
        if self.var_name == "theta":
            tasks = {}
            required_fields = ["t", "p"]
            for v in required_fields:
                tasks[v] = ERA5FieldCropped(
                    scene_id=self.scene_id,
                    var_name=v,
                    bbox=self.bbox,
                    datapath=self.datapath,
                )
            return tasks
        elif self.var_name in DERIVED_VARIABLES:
            tasks = {}
            required_fields = DERIVED_VARIABLES[self.var_name][1]
            for v in required_fields:
                tasks[v] = ERA5FieldCropped(
                    scene_id=self.scene_id,
                    var_name=v,
                    bbox=self.bbox,
                    datapath=self.datapath,
                )
            return tasks
        else:
            var_name = self.var_name
            if var_name == "sp":
                var_name = "lnsp"
            return ERA5Field(scene_id=self.scene_id, var_name=var_name)

    def run(self):
        if self.var_name in ["alt", "p"]:
            ds_cropped = self.input().open()
            da_cropped = ds_cropped[self.var_name]
        elif self.var_name == "theta":
            ds_cropped = xr.merge([inp.open() for inp in self.input().values()])
            da_cropped = atmos_variables.potential_temperature(
                ds=ds_cropped, vertical_coord="level"
            )
        elif self.var_name in DERIVED_VARIABLES:
            _field_func = DERIVED_VARIABLES[self.var_name][0]
            kwargs = {f"da_{v}": input.open() for (v, input) in self.input().items()}
            da_cropped = _field_func(**kwargs)
        else:
            da = self.input().open()
            da = da.rename(dict(latitude="lat", longitude="lon"))
            levels = slice(100, None)  # levels are in reverse order
            if "level" in da.coords:
                da = da.sel(level=levels)
            da_cropped = crop_field(da, self.bbox)

            if self.var_name == "sp":
                # what we actually loaded was `lnsp`
                da_cropped = np.exp(da_cropped)
                da_cropped.name = "sp"

        # ensure the time dimension is dropped so that we can stack on the
        # scene id later
        da_cropped = da_cropped.squeeze()

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_cropped.to_netcdf(str(self.output().fn))

    def output(self):
        fn = f"{self.scene_id}__{self.var_name}__cropped.nc"
        p = Path(self.datapath) / fn
        return utils.XArrayTarget(str(p))


class ERA5HeightAndPressureFieldCropped(luigi.Task):
    scene_id = luigi.Parameter()
    bbox = utils.BBoxParameter()
    datapath = luigi.Parameter()

    def requires(self):
        tasks = {}
        for v in ["q", "t", "z", "sp"]:
            tasks[v] = ERA5FieldCropped(
                scene_id=self.scene_id,
                bbox=self.bbox,
                var_name=v,
                datapath=self.datapath,
            )
        return tasks

    def run(self):
        dataarrays = [inp.open() for inp in self.input().values()]
        ds = xr.merge(dataarrays)
        ds_aux = calculate_heights_and_pressures(ds=ds)
        ds = ds_aux[["height_f", "p_f"]].rename(dict(height_f="alt", p_f="p"))
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        ds.to_netcdf(str(self.output().fn))

    def output(self):
        fn = f"{self.scene_id}__z_and_p__cropped.nc"
        p = Path(self.datapath) / fn
        return utils.XArrayTarget(str(p))
