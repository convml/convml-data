from pathlib import Path

import joblib
import luigi
import xarray as xr
from convml_tt.data.dataset import ImageSingletDataset
from convml_tt.data.transforms import get_transforms as get_model_transforms
from convml_tt.interpretation.embedding_transforms import apply_transform
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings

from ...utils.luigi import XArrayTarget
from ..tiles import SceneTilesData
from ..utils import SceneBulkProcessingBaseTask
from .defaults import PREDICTION_BATCH_SIZE
from .rect.sampling import (  # noqa
    SLIDING_WINDOW_EMBEDDINGS_DEFAULT_KWARGS,
    SceneSlidingWindowImageEmbeddings,
)
from .sampling_base import make_embedding_name


class _SceneTileEmbeddingsBase(luigi.Task):
    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    model_path = luigi.Parameter()
    prediction_batch_size = luigi.IntParameter(default=PREDICTION_BATCH_SIZE)

    def requires(self):
        fp_model = Path(self.model_path)
        fp_model_expected = Path(self.data_path) / "embedding_models"
        if not fp_model.parent.absolute() != fp_model_expected.absolute():
            raise Exception(f"embedding models should be stored in {fp_model_expected}")

        return SceneTilesData(
            data_path=self.data_path,
            scene_id=self.scene_id,
            tiles_kind=self.tiles_kind,
        )

    def _get_tile_dataset(self, model_transforms):
        raise NotImplementedError

    def run(self):
        model = TripletTrainerModel.load_from_checkpoint(self.model_path)
        model.freeze()

        model_transforms = get_model_transforms(
            step="predict", normalize_for_arch=model.base_arch
        )

        tile_dataset = self._get_tile_dataset(model_transforms=model_transforms)
        if len(tile_dataset) == 0:
            da_pred = xr.DataArray()
        else:

            da_pred = get_embeddings(
                model=model,
                tile_dataset=tile_dataset,
                prediction_batch_size=self.prediction_batch_size,
            )
            da_pred["triplet_tile_id"] = tile_dataset.df_tiles.triplet_tile_id

        da_pred.name = "emb"
        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        da_pred.to_netcdf(self.output().path)

    @property
    def tiles_kind(self):
        raise NotImplementedError

    def output(self):
        emb_name = make_embedding_name(model_path=self.model_path, kind=self.tiles_kind)
        path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name
        fn_out = f"{self.scene_id}.nc"

        return XArrayTarget(str(path / fn_out))


class SceneTripletTileEmbeddings(_SceneTileEmbeddingsBase):
    """
    Get embeddings for all tiles of a specific type (anchor, neighbor, distant) in a specific scene
    """

    tile_type = luigi.Parameter()
    tiles_kind = "triplets"
    dataset_stage = luigi.OptionalStrParameter(default="train")

    def _get_tile_dataset(self, model_transforms):
        first_tile = next(iter(self.input().values()))
        tiles_path = Path(first_tile["image"].path).parent.parent

        # read in the scene tile locations so that we can pick out the tiles
        # that belong to the requested dataset stage (train vs study).
        # We can't simply use the aux-field tile_id values because these
        # include all tiles in a scene (train or study)
        scene_tile_locations = (
            self.requires().requires()["tile_locations"].output().open()
        )
        scene_tile_ids = [
            f"{stl['triplet_id']:05d}_{stl['tile_type']}"
            for stl in scene_tile_locations
            if stl["triplet_collection"] == self.dataset_stage
        ]

        tile_type = self.tile_type

        tile_dataset = ImageSingletDataset(
            data_dir=tiles_path,
            transform=model_transforms,
            tile_type=tile_type.upper(),
            stage=self.dataset_stage,
        )

        # remove all but the tiles from the single scene we're interested in
        df_tiles = tile_dataset.df_tiles
        df_tiles["triplet_tile_id"] = df_tiles.apply(
            lambda row: f"{row.name:05d}_{row.tile_type}", axis=1
        )
        df_tiles_scene = df_tiles[df_tiles.triplet_tile_id.isin(scene_tile_ids)]
        tile_dataset.df_tiles = df_tiles_scene

        return tile_dataset

    def output(self):
        emb_name = make_embedding_name(
            model_path=self.model_path, kind=self.tiles_kind, stage=self.dataset_stage
        )
        path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name
        fn_out = f"{self.scene_id}.{self.tile_type}.nc"

        return XArrayTarget(str(path / fn_out))


class SceneTileEmbeddings(luigi.Task):
    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    model_path = luigi.Parameter()
    model_args = luigi.DictParameter(default={})

    def requires(self):
        fp_model = Path(self.model_path)
        fp_model_expected = Path(self.data_path) / "embedding_models"
        if not fp_model.parent.absolute() != fp_model_expected.absolute():
            raise Exception(f"embedding models should be stored in {fp_model_expected}")

        if self.tiles_kind == "rect-slidingwindow":
            return SceneSlidingWindowImageEmbeddings(
                data_path=self.data_path,
                scene_id=self.scene_id,
                model_path=self.model_path,
                **dict(self.model_args),
            )
        elif self.tiles_kind == "triplets":
            tasks = {}
            if "tile_type" not in self.model_args:
                tile_types = ["anchor", "neighbor", "distant"]
            else:
                tile_types = [self.model_args["tile_type"]]

            for tile_type in tile_types:
                tasks[tile_type] = SceneTripletTileEmbeddings(
                    data_path=self.data_path,
                    scene_id=self.scene_id,
                    model_path=self.model_path,
                    tile_type=tile_type,
                    **dict(self.model_args),
                )
            return tasks
        else:
            raise NotImplementedError(self.tiles_kind)

    def run(self):
        if self.tiles_kind == "triplets":
            datasets = []
            for tile_type, inp in self.input().items():
                da_embs_tiletype = inp.open()
                if da_embs_tiletype.count() == 0:
                    # scene doesn't contain any tiles
                    continue
                da_embs_tiletype = da_embs_tiletype.swap_dims(tile_id="triplet_tile_id")
                # put the tile-type into a coord so we can get a value for each tile
                da_embs_tiletype["tile_type"] = da_embs_tiletype.attrs.pop(
                    "tile_type"
                ).lower()
                datasets.append(da_embs_tiletype)

            if len(datasets) > 0:
                da = xr.concat(datasets, dim="triplet_tile_id")
            else:
                da = xr.DataArray(name="emb")
            da.to_netcdf(self.output().path)

    def output(self):
        if self.tiles_kind == "rect-slidingwindow":
            return self.input()
        else:
            model_args = dict(self.model_args)
            if self.tiles_kind == "triplets":
                # remove the tile_type so that it doesn't become part of the
                # directory name
                tile_type = model_args.pop("tile_type", "all")
                if tile_type != "all":
                    # if we're requesting a specific tile type then return the
                    # parent task
                    return self.input()[tile_type].output()

            emb_name = make_embedding_name(
                model_path=self.model_path, kind=self.tiles_kind, **dict(model_args)
            )
            path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name
            fn_out = f"{self.scene_id}.nc"

            return XArrayTarget(str(path / fn_out))


class DatasetScenesTileEmbeddings(SceneBulkProcessingBaseTask):
    """
    Produce embeddings for all tiles across all scenes in dataset
    """

    TaskClass = SceneTileEmbeddings

    tiles_kind = luigi.Parameter()
    data_path = luigi.Parameter(default=".")

    model_path = luigi.Parameter()
    model_args = luigi.DictParameter(default={})

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            tiles_kind=self.tiles_kind,
            model_path=self.model_path,
            model_args=self.model_args,
        )


def make_transform_name(transform, **transform_args):
    if "pretrained_model" in transform_args:
        return transform_args["pretrained_model"]

    if transform is None:
        return None

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


class _AggregatedTileEmbeddingsTransformMixin:
    embedding_transform = luigi.Parameter(default=None)
    embedding_transform_args = luigi.DictParameter(default={})

    def _apply_transform(self, da_emb):
        transform_model_args = dict(self.embedding_transform_args)

        if "pretrained_model" in transform_model_args:
            fp_pretrained_model = (
                Path(self.data_path)
                / "embeddings"
                / "models"
                / "{}.joblib".format(transform_model_args.pop("pretrained_model"))
            )
            transform_model_args["pretrained_model"] = joblib.load(fp_pretrained_model)

        da_emb, transform_model = apply_transform(
            da=da_emb,
            transform_type=self.embedding_transform,
            return_model=True,
            n_jobs=-1,  # default to using all CPU cores
            **transform_model_args,
        )

        if "pretrained_model" not in transform_model_args:
            if self.__class__.__module__ == "convml_data.pipeline.embeddings.sampling":
                embedding_model_path = self.model_path
                embedding_model_args = self.model_args
            else:
                embedding_model_path = self.embedding_model_path
                embedding_model_args = self.embedding_model_args

            emb_model_name = make_embedding_name(
                kind=self.tiles_kind,
                model_path=embedding_model_path,
                **embedding_model_args,
            )

            fn = f"{emb_model_name}.{self.transform_name}.joblib"
            fp_transform_model = Path(embedding_model_path).parent / fn
            joblib.dump(transform_model, fp_transform_model)

        return da_emb

    @property
    def transform_name(self):
        return make_transform_name(
            transform=self.embedding_transform,
            **dict(self.embedding_transform_args),
        )


class AggregatedDatasetScenesTileEmbeddings(
    SceneBulkProcessingBaseTask, _AggregatedTileEmbeddingsTransformMixin
):
    """
    Combine into a single file all embeddings for all tiles across all scenes in dataset
    """

    tiles_kind = luigi.Parameter()
    data_path = luigi.Parameter(default=".")

    model_path = luigi.Parameter()
    model_args = luigi.DictParameter(default={})

    def requires(self):
        kwargs = dict(
            tiles_kind=self.tiles_kind,
            data_path=self.data_path,
            model_path=self.model_path,
            model_args=self.model_args,
        )

        if self.embedding_transform is None:
            return DatasetScenesTileEmbeddings(**kwargs)
        else:
            return self.__class__(**kwargs)

    def run(self):
        if self.embedding_transform is None:
            datasets = []
            if self.tiles_kind == "rect-slidingwindow":
                concat_dim = "scene_id"
            elif self.tiles_kind == "triplets":
                # not all scenes contain the same number of triplets so we need
                # to stack by something else otherwise we'll get nan-values
                concat_dim = "triplet_tile_id"
            else:
                raise NotImplementedError(self.tiles_kind)

            datasets = []
            for scene_id, inp in self.input().items():
                da_scene = inp.open()
                if da_scene.count() == 0:
                    continue
                da_scene["scene_id"] = scene_id
                datasets.append(da_scene)
            da_all_scenes = xr.concat(datasets, dim=concat_dim)
        else:
            da_all_scenes = self.input().open()
            da_all_scenes = self._apply_transform(da_emb=da_all_scenes)

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        da_all_scenes.to_netcdf(self.output().path)

    def output(self):
        model_args = dict(self.model_args)
        if self.tiles_kind == "triplets":
            # remove the tile_type so that it doesn't become part of the
            # directory name
            tile_type = model_args.pop("tile_type", "all")
            if tile_type != "all":
                # if we're requesting a specific tile type then return the
                # parent task
                return self.input()[tile_type].output()

        emb_name = make_embedding_name(
            model_path=self.model_path, kind=self.tiles_kind, **dict(model_args)
        )
        path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name

        name_parts = ["__all__"]

        transform_name = self.transform_name
        if transform_name is not None:
            name_parts.append(transform_name)

        fn_out = f"{'.'.join(name_parts)}.nc"

        return XArrayTarget(str(path / fn_out))
