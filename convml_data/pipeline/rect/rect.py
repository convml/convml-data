#!/usr/bin/env python
# coding: utf-8
"""
Steps in processing:

    1. Load field from era5
    2. Crop to domain
    3. Calculate height field if it is missing
    4. Interpolate field (and heights) to correct grid
"""

from pathlib import Path

import luigi
import xarray as xr
import xesmf as xe

from ... import sources, utils


def find_bbox(da_grid, deg_pad=1.0):
    bbox_domain = dict(
        lon=(
            da_grid.lon.min().values - deg_pad,
            da_grid.lon.max().values + deg_pad,
        ),
        lat=(
            da_grid.lat.min().values - deg_pad,
            da_grid.lat.max().values + deg_pad,
        ),
    )
    return bbox_domain


class FieldReprojected(luigi.Task):
    scene_id = luigi.Parameter()
    var_name = luigi.Parameter()
    dst_grid_filename = luigi.Parameter()

    def _get_grid(self):
        da_emb = xr.open_dataarray(self.dst_grid_filename)
        return da_emb

    def requires(self):
        deg_pad = 1.0
        da_grid = self._get_grid()

        rad_vars = [
            "broadband_longwave_flux",
            "broadband_shortwave_albedo",
            "cloud_top_temperature",
        ]

        bbox_domain = find_bbox(da_grid=da_grid, deg_pad=deg_pad)

        if any([self.var_name.startswith(rad_var) for rad_var in rad_vars]):
            TaskClass = sources.ceres.CeresFieldCropped
        else:
            TaskClass = sources.era5.ERA5FieldCropped

        return TaskClass(
            scene_id=self.scene_id, bbox=bbox_domain, var_name=self.var_name
        )

    def run(self):
        da_grid = self._get_grid()
        da = self.input().open()
        regridder = xe.Regridder(
            da,
            da_grid,
            "bilinear",
        )

        da_rect = regridder(da)
        da_rect["x"] = da_grid.x
        da_rect["y"] = da_grid.y
        # ensure units are copied across
        da_rect.attrs.update(da.attrs)
        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_rect.to_netcdf(str(self.output().fn))

    def output(self):
        fn = f"{self.scene_id}__{self.var_name}__reproj.nc"
        p = utils.DATA_PATH / fn
        return utils.XArrayTarget(str(p))


class AllScenes(luigi.WrapperTask):
    dst_grid_filename = luigi.Parameter()

    def _get_grid(self):
        da_emb = xr.open_dataarray(self.dst_grid_filename)
        return da_emb

    def requires(self):
        da_grid = self._get_grid()

        tasks = {}
        for scene_id in da_grid.scene_id.values:
            scene_tasks = []

            era5_vars = ["q", "u", "v", "t", "alt", "p", "theta"]
            rad_vars = [
                "broadband_longwave_flux__goes16n",
                # "broadband_longwave_flux__meteosat9n",
                "broadband_shortwave_albedo__goes16n",
                # "broadband_shortwave_albedo__meteosat9n",
                "cloud_top_temperature__goes16n",
            ]
            for v in era5_vars + rad_vars:
                t = FieldReprojected(
                    scene_id=scene_id,
                    var_name=v,
                    dst_grid_filename=self.dst_grid_filename,
                )
                scene_tasks.append(t)
            tasks[scene_id] = scene_tasks
        return tasks
