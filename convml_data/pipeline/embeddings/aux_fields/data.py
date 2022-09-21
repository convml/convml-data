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

import luigi
import regridcart as rc
import xarray as xr
import xesmf as xe

from ...aux_sources import CheckForAuxiliaryFiles
from ...embeddings.rect.sampling import SceneSlidingWindowImageEmbeddings
from ...embeddings.sampling import (
    SceneTileEmbeddings,
    _AggregatedTileEmbeddingsTransformMixin,
    make_embedding_name,
    make_transform_name,
)
from ...sampling import CropSceneSourceFiles
from ...tiles import SceneTilesData
from ...utils import SceneBulkProcessingBaseTask
from . import data_filters, utils


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
    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    aux_name = luigi.Parameter()

    def _parse_filters(self):
        if self.filters is not None:
            return data_filters.parse_defs(self.filters)
        return []

    def requires(self):
        tasks = {}
        tasks["emb"] = SceneSlidingWindowImageEmbeddings(
            data_path=self.data_path,
            scene_id=self.scene_id,
            model_path=self.embedding_model_path,
            **self.embedding_model_args,
        )

        # aux field tasks below
        tasks["aux"] = CropSceneSourceFiles(
            scene_id=self.scene_id,
            data_path=self.data_path,
            aux_name=self.aux_name,
        )

        return tasks

    def reduce(self, da):
        try:
            return getattr(da, self.reduce_op)()
        except AttributeError:
            raise NotImplementedError(self.reduce_op)

    def run(self):
        da_scalar = self.input()["aux"]["data"].open()
        da_emb_src_scene = self.input()["emb"].open()

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

        # ensure that the scalar has the name that we requested data for
        da_scalar.name = self.aux_name

        ds = xr.merge([da_scalar, da_emb])

        Path(self.output().fn).parent.mkdir(exist_ok=True)
        ds.to_netcdf(self.output().fn)

    def output(self):
        emb_name = make_embedding_name(
            kind="rect-slidingwindow",
            model_path=self.embedding_model_path,
            **self.embedding_model_args,
        )

        name_parts = [
            self.scene_id,
            self.aux_name,
            "by",
            emb_name,
        ]

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
        if self.tiles_kind == "rect-slidingwindow":
            return SceneSlidingWindowEmbeddingsAuxFieldRegridding(
                data_path=self.data_path,
                scene_id=self.scene_id,
                aux_name=self.aux_name,
                embedding_model_path=self.embedding_model_path,
                embedding_model_args=self.embedding_model_args,
            )
        else:
            tasks = {}
            tasks["aux"] = SceneTilesData(
                data_path=self.data_path,
                aux_name=self.aux_name,
                scene_id=self.scene_id,
                tiles_kind=self.tiles_kind,
            )
            tasks["embeddings"] = SceneTileEmbeddings(
                data_path=self.data_path,
                scene_id=self.scene_id,
                tiles_kind=self.tiles_kind,
                model_path=self.embedding_model_path,
                model_args=self.embedding_model_args,
            )
            return tasks

    def run(self):
        if self.tiles_kind == "rect-slidingwindow":
            pass
        elif self.tiles_kind == "triplets":
            aux_fields = self.input()["aux"]
            da_embs = self.input()["embeddings"].open()

            if da_embs.count() == 0:
                ds = xr.Dataset()
            else:
                values = []
                for triplet_tile_id in da_embs.triplet_tile_id.values:
                    da_tile_aux = aux_fields[triplet_tile_id]["data"].open()
                    da_tile_aux_value = da_tile_aux.mean(keep_attrs=True)
                    da_tile_aux_value.attrs["long_name"] = (
                        "tile mean " + da_tile_aux_value.long_name
                    )
                    da_tile_aux_value["triplet_tile_id"] = triplet_tile_id
                    values.append(da_tile_aux_value)

                da_aux_values = xr.concat(values, dim="triplet_tile_id")
                ds = xr.merge([da_embs, da_aux_values])

            Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
            ds.to_netcdf(self.output().path)
        else:
            raise NotImplementedError(self.tiles_kind)

    def output(self):
        if self.tiles_kind == "rect-slidingwindow":
            return self.input()

        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_path=self.embedding_model_path,
            **dict(self.embedding_model_args),
        )

        name_parts = [
            self.scene_id,
            self.aux_name,
            "by",
            emb_name,
        ]

        fn = ".".join(name_parts) + ".nc"

        p = Path(self.data_path) / "embeddings" / "with_aux" / self.tiles_kind / fn
        return utils.XArrayTarget(str(p))


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


class AggregatedDatasetScenesAuxFieldWithEmbeddings(
    luigi.Task, _AggregatedTileEmbeddingsTransformMixin
):
    """
    Combine across all scenes in a dataset the embeddings and aux-field values
    into a single file. Also, optionally apply an embedding transform
    """

    aux_name = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

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
            if self.tiles_kind == "rect-slidingwindow":
                concat_dim = "scene_id"
            elif self.tiles_kind == "triplets":
                concat_dim = "triplet_tile_id"
            else:
                raise NotImplementedError(self.tiles_kind)

            for scene_id, inp in self.input().items():
                ds_scene = inp.open()
                if len(ds_scene.data_vars) == 0:
                    # no tiles in this scene
                    continue

                ds_scene["scene_id"] = scene_id
                datasets.append(ds_scene)

            ds = xr.concat(datasets, dim=concat_dim)
            if "stage" in ds.data_vars:
                ds = ds.set_coords(["stage", "scene_id"])
            else:
                ds = ds.set_coords(["scene_id"])
        else:
            ds = self.input().open()
            # if "pretrained_model" in self.embedding_model_args:
            # datasets = []
            # da_embs_scene
            # else:
            ds["emb"] = self._apply_transform(da_emb=ds.emb)

        ds.to_netcdf(self.output().path)

    @property
    def transform_name(self):
        return make_transform_name(
            transform=self.embedding_transform,
            aux=self.aux_name,
            **dict(self.embedding_transform_args),
        )

    def output(self):
        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_path=self.embedding_model_path,
            **dict(self.embedding_model_args),
        )

        name_parts = [
            "__all__",
            self.aux_name,
            "by",
            emb_name,
        ]

        transform_name = self.transform_name

        if transform_name is not None:
            name_parts.append(transform_name)

        name = ".".join(name_parts)
        p = Path(self.data_path) / "embeddings" / "with_aux"
        return utils.XArrayTarget(str(p / (name + ".nc")))
