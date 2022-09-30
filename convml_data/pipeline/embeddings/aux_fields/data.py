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
import xarray as xr

from ...aux_sources import CheckForAuxiliaryFiles
from ...embeddings.sampling import (
    SceneTileEmbeddings,
    _AggregatedTileEmbeddingsTransformMixin,
    make_embedding_name,
    make_transform_name,
)
from ...tiles import SceneTilesData
from ...utils import SceneBulkProcessingBaseTask
from . import utils


class SceneAuxFieldWithEmbeddings(luigi.Task):
    aux_name = luigi.OptionalParameter()
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    data_path = luigi.Parameter(default=".")
    crop_pad_ptc = luigi.FloatParameter(default=0.1)
    create_aux_tile_images = luigi.BoolParameter(default=False)

    def requires(self):
        tasks = {}
        aux_kwargs = {}
        if self.tiles_kind == "rect-slidingwindow":
            aux_kwargs["step_size"] = self.embedding_model_args["step_size"]

        tasks["aux"] = SceneTilesData(
            data_path=self.data_path,
            aux_name=self.aux_name,
            scene_id=self.scene_id,
            tiles_kind=self.tiles_kind,
            extra_args=aux_kwargs,
            create_images=self.create_aux_tile_images,
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
        aux_fields = self.input()["aux"]
        da_embs = self.input()["embeddings"].open()

        if da_embs.count() == 0:
            ds = xr.Dataset()
        else:
            if self.tiles_kind == "triplets":
                concat_dim = "triplet_tile_id"
            else:
                concat_dim = "tile_id"
                da_embs = da_embs.stack(n=("x", "y")).swap_dims(n="tile_id").drop("n")

            values = []

            for t_id in da_embs[concat_dim].values:
                da_tile_aux = aux_fields[t_id]["data"].open()
                da_tile_aux_value = da_tile_aux.mean(keep_attrs=True)
                da_tile_aux_value.attrs["long_name"] = (
                    "tile mean " + da_tile_aux_value.long_name
                )
                da_tile_aux_value[concat_dim] = t_id
                values.append(da_tile_aux_value)

            da_aux_values = xr.concat(values, dim=concat_dim)
            ds = xr.merge([da_embs, da_aux_values])

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        ds.to_netcdf(self.output().path)

    def output(self):
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
            if self.tiles_kind in ["trajectories", "rect-slidingwindow"]:
                concat_dim = "tile_id"
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
