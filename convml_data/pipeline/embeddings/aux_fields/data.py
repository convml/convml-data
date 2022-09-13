"""
Produce co-located values of a) cloud organising embeddings and b) a
auxiliary scalar field. There are a few ways this co-location is done:

1. Individual tiles onto which the auxiliary field is regridded and the
   auxiliary is then reduced with a specific operation (for example computing
   the mean)
2. `rect` regridded domains on which embeddings are produced at a fixed spacing
   by sampling tiles on this domain. This produces embedding vectors at
   individual points. These embeddings are then interpolated onto the grid of
   the auxiliary fields.
"""
from pathlib import Path

import joblib
import luigi
import numpy as np
import regridcart as rc
import scipy.ndimage.filters
import xarray as xr
import xesmf as xe
from convml_tt.interpretation.embedding_transforms import apply_transform

from ...aux_sources import CheckForAuxiliaryFiles
from ...embeddings.rect.sampling import (
    SceneSlidingWindowImageEmbeddings,
    make_embedding_name,
)
from ...embeddings.sampling import TileEmbeddings, model_identifier_from_filename
from ...sampling import CropSceneSourceFiles
from ...tiles import SceneTilesData
from ...utils import SceneBulkProcessingBaseTask
from . import data_filters, utils


def make_transform_name(transform, transform_args={}):
    skip_transform_args = []
    name_parts = [
        transform,
    ]

    if len(transform_args) > 0:
        name_parts += [
            f"{k}__{v}"
            for (k, v) in transform_args.items()
            if k not in skip_transform_args
        ]
    return ".".join(name_parts)


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


class SceneSlidingWindowEmbeddingsAuxFieldRegridding(luigi.Task):
    """
    Regrid embedding variable onto grid of auxillariary field for a single
    scene and optionally reduce using a given operation per segment
    """

    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.IntParameter(default=10)
    prediction_batch_size = luigi.IntParameter(default=32)

    aux_name = luigi.Parameter()

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
        tasks = {}
        tasks["emb"] = SceneSlidingWindowImageEmbeddings(
            data_path=self.data_path,
            scene_id=self.scene_id,
            model_path=self.model_path,
            step_size=self.step_size,
            prediction_batch_size=self.prediction_batch_size,
        )

        # aux field tasks below
        tasks["aux"] = CropSceneSourceFiles(
            scene_id=self.scene_id,
            data_path=self.data_path,
            aux_name=self.aux_name,
        )
        for filter_meta in self._parse_filters():
            filter_tasks = {}
            for reqd_field_name in filter_meta["reqd_props"]:
                filter_tasks[reqd_field_name] = CropSceneSourceFiles(
                    scene_id=self.scene_id,
                    data_path=self.data_path,
                    aux_name=reqd_field_name,
                )
            tasks.setdefault("filter_data", []).append(filter_tasks)

        return tasks

    def reduce(self, da):
        try:
            return getattr(da, self.reduce_op)()
        except AttributeError:
            raise NotImplementedError(self.reduce_op)

    def run(self):
        da_scalar = self.input()["aux"]["data"].open()
        da_emb_src_scene = self.input()["emb"].open()

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
            da_segments_src["lat"] = da_emb_src_scene.lat
            da_segments_src["lon"] = da_emb_src_scene.lon
            da_segments_src = da_segments_src.sel(scene_id=self.scene_id)

            # regrid segments onto scalar's grid
            da_segments = regrid_emb_var_onto_variable(
                da_emb=da_segments_src, da_v=da_scalar
            )
        else:
            da_segments = None

        # regrid embedding position onto scalar xy-grid (lat,lon)
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
        da_scalar.name = self.aux_name

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
        model_name = model_identifier_from_filename(fn=Path(self.model_path).name)
        emb_name = f"{model_name}.{self.step_size}"

        name_parts = [
            self.scene_id,
            self.aux_name,
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

        p = Path(self.data_path) / "embeddings" / "with_aux" / fn
        return utils.XArrayTarget(str(p))


class SceneAuxFieldWithEmbeddings(luigi.Task):
    aux_name = luigi.OptionalParameter()
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    data_path = luigi.Parameter(default=".")
    crop_pad_ptc = luigi.FloatParameter(default=0.1)

    def requires(self):
        prediction_batch_size = self.embedding_model_args.get(
            "prediction_batch_size", 32
        )
        step_size = self.embedding_model_args.get("step_size", 100)

        if self.tiles_kind == "rect-slidingwindow":
            return SceneSlidingWindowEmbeddingsAuxFieldRegridding(
                data_path=self.data_path,
                scene_id=self.scene_id,
                aux_name=self.aux_name,
                model_path=self.embedding_model_path,
                step_size=step_size,
                prediction_batch_size=prediction_batch_size,
            )
        else:
            tasks = {}
            tasks["aux"] = SceneTilesData(
                data_path=self.data_path,
                aux_name=self.aux_name,
                scene_id=self.scene_id,
                tiles_kind=self.tiles_kind,
            )
            tasks["embeddings"] = TileEmbeddings(
                data_path=self.data_path,
                scene_id=self.scene_id,
                tiles_kind=self.tiles_kind,
                model_path=self.embedding_model_path,
                step_size=step_size,
                prediction_batch_size=prediction_batch_size,
            )
            return tasks

    def run(self):
        if self.tiles_kind == "rect-slidingwindow":
            pass
        else:
            raise NotImplementedError(self.tiles_kind)

    def output(self):
        if self.tiles_kind == "rect-slidingwindow":
            return self.input()
        else:
            raise NotImplementedError(self.tiles_kind)


class DatasetScenesAuxFieldWithEmbeddings(SceneBulkProcessingBaseTask):
    TaskClass = SceneAuxFieldWithEmbeddings
    SceneIDsTaskClass = CheckForAuxiliaryFiles

    aux_name = luigi.OptionalParameter()
    tiles_kind = luigi.Parameter()

    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    data_path = luigi.Parameter(default=".")
    crop_pad_ptc = luigi.FloatParameter(default=0.1)

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            aux_name=self.aux_name,
            tiles_kind=self.tiles_kind,
            embedding_model_path=self.embedding_model_path,
            embedding_model_args=self.embedding_model_args,
        )

    def _get_scene_ids_task_kwargs(self):
        return dict(aux_name=self.aux_name)


class AggregatedDatasetScenesAuxFieldWithEmbeddings(luigi.Task):
    """
    Combine across all scenes in a dataset the embeddings and aux-field values
    into a single file. Also, optionally apply an embedding transform
    """

    aux_name = luigi.Parameter()
    scene_ids = luigi.OptionalParameter(default=[])
    tiles_kind = luigi.Parameter()

    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})
    embedding_transform = luigi.Parameter(default=None)
    embedding_transform_args = luigi.DictParameter(default={})

    data_path = luigi.Parameter(default=".")

    def requires(self):
        kwargs = dict(
            data_path=self.data_path,
            aux_name=self.aux_name,
            tiles_kind=self.tiles_kind,
            embedding_model_path=self.embedding_model_path,
            embedding_model_args=self.embedding_model_args,
        )
        if self.embedding_transform is None:
            return DatasetScenesAuxFieldWithEmbeddings(**kwargs)

        return AggregatedDatasetScenesAuxFieldWithEmbeddings(**kwargs)

    def run(self):
        if self.embedding_transform is None:
            datasets = []
            for scene_id, inp in self.input().items():
                ds_scene = inp.open()
                ds_scene["scene_id"] = scene_id
                datasets.append(ds_scene)
            ds = xr.concat(datasets, dim="scene_id")
        else:
            ds = self.input().open()
            transform_model_args = dict(self.embedding_transform_args)

            if "pretrained_model" in transform_model_args:
                fp_pretrained_model = (
                    Path(self.data_path)
                    / "embeddings"
                    / "models"
                    / "{}.joblib".format(transform_model_args.pop("pretrained_model"))
                )
                transform_model_args["pretrained_model"] = joblib.load(
                    fp_pretrained_model
                )

            ds["emb"], transform_model = apply_transform(
                da=ds.emb,
                transform_type=self.embedding_transform,
                return_model=True,
                **transform_model_args,
            )

            if "pretrained_model" not in transform_model_args:
                emb_model_name = make_embedding_name(
                    kind=self.tiles_kind,
                    model_path=self.embedding_model_path,
                    model_args=self.embedding_model_args,
                )
                transform_name = make_transform_name(
                    transform=self.embedding_transform,
                    transform_args=self.embedding_transform_args,
                )
                fn = f"{emb_model_name}.{transform_name}.joblib"
                fp_transform_model = Path(self.embedding_model_path).parent / fn
                joblib.dump(transform_model, fp_transform_model)

        ds.to_netcdf(self.output().path)

    def output(self):
        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_path=self.embedding_model_path,
            model_args=self.embedding_model_args,
        )

        name_parts = [
            "__all__",
            self.aux_name,
            "by",
            emb_name,
        ]

        if self.embedding_transform is not None:
            transform_model_args = dict(self.embedding_transform_args)
            if "pretrained_model" in transform_model_args:
                transform_name = transform_model_args["pretrained_model"]
            else:
                transform_name = make_transform_name(
                    transform=self.embedding_transform,
                    transform_args=self.embedding_transform_args,
                )
            name_parts.append(transform_name)

        fn = ".".join(name_parts) + ".nc"

        p = Path(self.data_path) / "embeddings" / "with_aux" / fn

        return utils.XArrayTarget(str(p))
