from pathlib import Path

import luigi
import numpy as np
import regridcart as rc
import scipy.ndimage.filters
import xarray as xr
import xesmf as xe

from . import data_filters, utils
from .datasources import FieldCropped
from .rect import find_bbox


def scale_km_to_m(da_c):
    assert da_c.units == "km"
    da_c = 1000 * da_c
    da_c.attrs["units"] = "m"
    return da_c


def regrid_emb_var_onto_variable(da_emb, da_v, method="nearest_s2d"):
    """
    Using lat,lon in `da_v` as target grid interpolate `da_emb` on this
    grid using lat,lon in `da_emb` as source grid
    """
    ds_grid_dst = da_v
    # ds_grid_dst = xr.Dataset()
    # ds_grid_dst["lat"] = da_v.lat  # .drop_vars(["lat", "lon"])
    # ds_grid_dst["lon"] = da_v.lon  # .drop_vars(["lat", "lon"])
    # ds_grid_dst

    ds_grid_src = xr.Dataset()
    ds_grid_src["lat"] = da_emb.lat.drop_vars(["lat", "lon"])
    ds_grid_src["lon"] = da_emb.lon.drop_vars(["lat", "lon"])

    regridder = xe.Regridder(
        ds_grid_src,
        ds_grid_dst,
        method,
    )

    if "emb_dim" in da_emb.dims:
        da_emb_regridded = da_emb.groupby("emb_dim").apply(regridder)
    else:
        da_emb_regridded = regridder(da_emb)

    da_emb_regridded.name = da_emb.name
    return da_emb_regridded


class ColumnScalarEmbeddingScatterData(luigi.Task):
    """
    Regrid embedding variable onto grid of auxillariary field for a single
    scene and optionally reduce using a given operation per segment
    """

    scalar_name = luigi.Parameter()
    scene_id = luigi.Parameter()
    emb_filepath = luigi.Parameter()
    datapath = luigi.Parameter()

    column_function = luigi.Parameter(default=None)
    segments_filepath = luigi.Parameter(default=None)
    segmentation_threshold = luigi.FloatParameter(default=0.0005)
    # n_scalar_bins = luigi.IntParameter(default=None)
    reduce_op = luigi.Parameter(default="mean")

    filters = luigi.OptionalParameter(default=None)

    def _parse_filters(self):
        if self.filters is not None:
            return data_filters.parse_defs(self.filters)
        return []

    def requires(self):
        da_emb = xr.open_dataarray(self.emb_filepath)
        deg_pad = 1.0
        bbox_domain = find_bbox(da_grid=da_emb, deg_pad=deg_pad)
        tasks = {}
        tasks["data"] = FieldCropped(
            scene_id=self.scene_id,
            bbox=bbox_domain,
            var_name=self.scalar_name,
            datapath=self.datapath,
        )
        for filter_meta in self._parse_filters():
            filter_tasks = {}
            for reqd_field_name in filter_meta["reqd_props"]:
                filter_tasks[reqd_field_name] = FieldCropped(
                    scene_id=self.scene_id,
                    bbox=bbox_domain,
                    var_name=reqd_field_name,
                    datapath=self.datapath,
                )
            tasks.setdefault("filter_data", []).append(filter_tasks)

        return tasks

    def reduce(self, da):
        try:
            return getattr(da, self.reduce_op)()
        except AttributeError:
            raise NotImplementedError(self.reduce_op)

    def run(self):
        da_scalar = self.input()["data"].open()
        da_emb_src = xr.open_dataarray(self.emb_filepath)

        filters = self._parse_filters()
        if len(filters) > 0:
            ds = xr.Dataset(
                dict(
                    data=da_scalar,
                )
            )
            for filter_meta, filter_input in zip(filters, self.input()["filter_data"]):
                for reqd_field_name in filter_meta["reqd_props"]:
                    da_filter_field = filter_input[reqd_field_name].open()
                    if "lat" not in da_filter_field.coords:
                        coords = rc.coords.get_latlon_coords_using_crs(
                            da=da_filter_field
                        )
                        da_filter_field["lat"] = coords["lat"]
                        da_filter_field["lon"] = coords["lon"]

                    da_filter_field_interpolated = regrid_emb_var_onto_variable(
                        da_filter_field, da_scalar
                    )
                    ds[reqd_field_name] = da_filter_field_interpolated

            ds = filter_meta["fn"](ds)

            da_scalar = ds.data

        if self.segments_filepath is not None:
            da_segments_src = xr.open_dataarray(self.segments_filepath)
            da_segments_src = da_segments_src.assign_coords(
                **{c: scale_km_to_m(da_segments_src[c]) for c in "xy"}
            )
            da_segments_src = da_segments_src.sel(k=self.segmentation_threshold)
            da_segments_src["lat"] = da_emb_src.lat
            da_segments_src["lon"] = da_emb_src.lon
            da_segments_src = da_segments_src.sel(scene_id=self.scene_id)

            # regrid segments onto scalar's grid
            da_segments = regrid_emb_var_onto_variable(
                da_emb=da_segments_src, da_v=da_scalar
            )
        else:
            da_segments = None

        # regrid embedding position onto scalar xy-grid (lat,lon)
        da_emb_src_scene = da_emb_src.sel(scene_id=self.scene_id)

        if "lat" not in da_scalar.coords:
            # and "lat" not in da_scalar.data_vars:
            coords = rc.coords.get_latlon_coords_using_crs(da=da_scalar)
            da_scalar["lat"] = coords["lat"]
            da_scalar["lon"] = coords["lon"]

        # the xy-coordinate values should come from the embeddings, so let's
        # get rid of them if they're on the scalar data
        for d in ["x", "y"]:
            if d in da_scalar:
                da_scalar = da_scalar.drop(d)

        da_emb = regrid_emb_var_onto_variable(da_emb=da_emb_src_scene, da_v=da_scalar)

        if self.column_function is not None:
            da_scalar = self._apply_column_function(da_scalar=da_scalar)

        dims = da_scalar.squeeze().dims
        if len(dims) > 2:
            raise Exception(
                "The scalar field should only have lat and lon coordinates"
                f" but it has `{dims}`. Please apply a `column_function`"
            )

        if "t" in da_scalar.coords and da_scalar.t != da_emb.t:
            da_dt = abs(da_scalar.t - da_emb.t)
            dt_seconds = da_dt.dt.seconds.item()
            if not dt_seconds < 10:
                raise Exception(da_dt.data, da_scalar.t.data, da_emb.squeeze().t.data)
            da_scalar = da_scalar.drop("t")

        # ensure that the scalar has the name that we requested data for
        da_scalar.name = self.scalar_name

        ds = xr.merge([da_scalar, da_emb])

        if da_segments is not None:
            ds = self.reduce(ds.groupby(da_segments))

        Path(self.output().fn).parent.mkdir(exist_ok=True)
        ds.to_netcdf(self.output().fn)

    def _apply_column_function(self, da_scalar):
        fns_names = self.column_function.split("__")

        for fn_name in fns_names:
            if fn_name.startswith("level_"):
                level_int = int(fn_name.split("_")[-1])
                da_scalar = da_scalar.sel(level=level_int)
            elif fn_name.startswith("leveldiff_"):
                level_a, level_b = fn_name.split("_")[1:]
                level_a = int(level_a)
                level_b = int(level_b)
                da_scalar_a = da_scalar.sel(level=level_a)
                da_scalar_b = da_scalar.sel(level=level_b)
                long_name = f"diff level {level_a}-{level_b} of {da_scalar.long_name}"
                units = da_scalar.units
                da_scalar = da_scalar_a - da_scalar_b
                da_scalar.attrs["long_name"] = long_name
                da_scalar.attrs["units"] = units
            elif fn_name.startswith("wavg_") or fn_name.startswith("wavgclip_"):
                # perform windowed horizontal averaging by applying an
                # averaging stencil of a given size
                Nk = int(fn_name.split("_")[-1])
                fn = scipy.ndimage.filters.convolve

                kernel = np.ones((Nk, Nk))
                kernel /= np.sum(kernel)

                old_attrs = da_scalar.attrs
                da_scalar = xr.apply_ufunc(fn, da_scalar, kwargs=dict(weights=kernel))
                da_scalar.attrs.update(old_attrs)
                da_scalar.attrs["long_name"] = f"window avg {da_scalar.long_name}"
                da_scalar.attrs["avg_window_size"] = Nk

                if fn_name.startswith("wavgclip_"):
                    N_clip = (Nk - 1) // 2
                    if N_clip != 0:
                        da_scalar = da_scalar.isel(
                            lat=slice(N_clip, -N_clip), lon=slice(N_clip, -N_clip)
                        )

            else:
                raise NotImplementedError(fn_name)

        return da_scalar

    def output(self):
        emb_name = Path(self.emb_filepath).name.replace(".nc", "")

        name_parts = [
            self.scene_id,
            self.scalar_name,
            "by",
            emb_name,
        ]

        if self.filters is not None:
            name_parts.append(self.filters.replace("=", ""))

        if self.column_function is not None:
            name_parts.append(self.column_function)

        if self.segments_filepath is not None:
            name_parts.append(f"{self.segmentation_threshold}seg")
            name_parts.append(self.reduce_op)

        fn = ".".join(name_parts) + ".nc"

        p = Path(self.datapath) / "scatterdata" / fn
        return utils.XArrayTarget(str(p))


class RegridEmbeddingsOnAuxField(luigi.Task):
    scalar_name = luigi.Parameter()
    scene_ids = luigi.ListParameter(default=[])
    emb_filepath = luigi.Parameter()
    datapath = luigi.Parameter()

    column_function = luigi.Parameter(default=None)
    segments_filepath = luigi.Parameter(default=None)
    segmentation_threshold = luigi.FloatParameter(default=0.0005)
    reduce_op = luigi.Parameter(default="mean")

    filters = luigi.OptionalParameter(default=None)

    def requires(self):
        if len(self.scene_ids) == 0:
            da_emb_src = xr.open_dataarray(self.emb_filepath)
            scene_ids = da_emb_src.scene_id.values
        else:
            scene_ids = self.scene_ids

        return {
            scene_id: ColumnScalarEmbeddingScatterData(
                scene_id=scene_id,
                scalar_name=self.scalar_name,
                emb_filepath=self.emb_filepath,
                column_function=self.column_function,
                segments_filepath=self.segments_filepath,
                segmentation_threshold=self.segmentation_threshold,
                # n_scalar_bins=self.n_scalar_bins,
                reduce_op=self.reduce_op,
                datapath=self.datapath,
                filters=self.filters,
            )
            for scene_id in scene_ids
        }

    def run(self):
        das = [inp.open() for inp in self.input().values()]

        # concat_dim = None
        # for d in ["time", "scene_id"]:
        # if d in das[0].dims:
        # concat_dim = d

        # if d is None:
        # raise Exception("Not sure what dimension to concat on")

        concat_dim = "scene_id"
        ds = xr.concat(das, dim=concat_dim)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        ds.to_netcdf(self.output().fn)

    def output(self):
        emb_name = Path(self.emb_filepath).name.replace(".nc", "")

        name_parts = [
            self.scalar_name,
            "regridded",
        ]

        if self.filters is not None:
            name_parts.append(self.filters.replace("=", ""))

        if self.column_function is not None:
            name_parts.append(self.column_function)

        if self.segments_filepath is not None:
            name_parts.append(f"{self.segmentation_threshold}seg")

        if self.segments_filepath is not None or self.column_function is not None:
            name_parts.append(self.reduce_op)

        identifier = ".".join(name_parts)
        fn = f"{identifier}.nc"

        p = Path(self.datapath) / emb_name / fn
        return utils.XArrayTarget(str(p))
