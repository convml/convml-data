from pathlib import Path

import joblib
import luigi
import luigi.util
import xarray as xr
from convml_tt.data.dataset import ImageSingletDataset
from convml_tt.data.transforms import get_transforms as get_model_transforms
from convml_tt.interpretation.embedding_transforms import apply_transform
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings

from ... import DataSource
from ...utils.luigi import XArrayTarget
from ..sampling import GenerateSceneIDs
from ..tiles import SceneTilesData
from ..triplets import TripletSceneSplits
from ..utils import SceneBulkProcessingBaseTask
from .defaults import PREDICTION_BATCH_SIZE
from .rect.sampling import SceneSlidingWindowImageEmbeddings  # noqa
from .sampling_base import _EmbeddingModelMixin, make_embedding_name


class _SceneTileEmbeddingsBase(luigi.Task, _EmbeddingModelMixin):
    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    prediction_batch_size = luigi.IntParameter(default=PREDICTION_BATCH_SIZE)

    def requires(self):
        fp_model = self.model_path
        if not fp_model.exists():
            raise Exception(f"Couldn't find model in path `{fp_model}`")

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
            # TODO: generalise for trajectory-based tiles
            da_pred["triplet_tile_id"] = tile_dataset.df_tiles.triplet_tile_id
            da_pred = da_pred.swap_dims(dict(tile_id="triplet_tile_id"))

        da_pred.name = "emb"
        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        da_pred.to_netcdf(self.output().path)

    @property
    def tiles_kind(self):
        raise NotImplementedError

    def output(self):
        emb_name = make_embedding_name(model_name=self.model_name, kind=self.tiles_kind)
        path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name
        fn_out = f"{self.scene_id}.nc"

        return XArrayTarget(str(path / fn_out))


class SceneTripletTileEmbeddings(_SceneTileEmbeddingsBase):
    """
    Get embeddings for all tiles of a specific type (anchor, neighbor, distant) in a specific scene
    """

    tile_type = luigi.Parameter()
    tiles_kind = "triplets"
    dataset_stage = luigi.Parameter()

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
        # remove all but the requested tile type
        scene_tile_locations = [
            stl for stl in scene_tile_locations if stl["tile_type"] == self.tile_type
        ]
        scene_tile_ids = [
            f"{stl['triplet_id']:05d}_{stl['tile_type']}"
            for stl in scene_tile_locations
            if stl["triplet_collection"] == self.dataset_stage
        ]

        if len(scene_tile_ids) == 0:
            # can return anything with length zero
            return []

        tile_type = self.tile_type

        def _include_only_scene_tiles(df_tiles):
            # remove all but the tiles from the single scene we're interested in
            df_tiles["triplet_tile_id"] = df_tiles.apply(
                lambda row: f"{row.name:05d}_{row.tile_type}", axis=1
            )
            df_tiles_scene = df_tiles[df_tiles.triplet_tile_id.isin(scene_tile_ids)]
            return df_tiles_scene

        tile_dataset = ImageSingletDataset(
            data_dir=tiles_path,
            transform=model_transforms,
            tile_type=tile_type.upper(),
            stage=self.dataset_stage,
            filter_func=_include_only_scene_tiles,
        )

        return tile_dataset

    def output(self):
        emb_name = make_embedding_name(
            model_name=self.model_name,
            kind=self.tiles_kind,
            dataset_stage=self.dataset_stage,
        )
        path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name
        fn_out = f"{self.scene_id}.{self.tile_type}.nc"

        return XArrayTarget(str(path / fn_out))


class SceneTileEmbeddings(luigi.Task, _EmbeddingModelMixin):
    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    model_args = luigi.DictParameter(default={})

    @property
    def datasource(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        if self.tiles_kind == "rect-slidingwindow":
            sampling_meta = self.datasource.sampling
            sampling_sw = sampling_meta.get("rect-slidingwindow")
            if sampling_sw is None or sampling_sw.get("tile_N") is None:
                raise Exception(
                    "Please define the tile-size, in pixels `tile_N` in "
                    " a section called `rect-slidingwindow` in `meta.yaml`"
                )
            N_tile = int(sampling_sw["tile_N"])

            return SceneSlidingWindowImageEmbeddings(
                data_path=self.data_path,
                scene_id=self.scene_id,
                model_name=self.model_name,
                N_tile=N_tile,
                **dict(self.model_args),
            )
        elif self.tiles_kind == "triplets":
            tasks = {}
            model_args = dict(self.model_args)
            if "tile_type" not in self.model_args:
                tile_types = ["anchor", "neighbor", "distant"]
            else:
                tile_types = [model_args.pop("tile_type")]

            for tile_type in tile_types:
                tasks[tile_type] = SceneTripletTileEmbeddings(
                    data_path=self.data_path,
                    scene_id=self.scene_id,
                    model_name=self.model_name,
                    tile_type=tile_type,
                    **model_args,
                )
            return tasks
        else:
            raise NotImplementedError(self.tiles_kind)

    def run(self):
        if self.tiles_kind == "triplets":
            # only requested one tile type, already produced the output file in
            # the parent task
            if len(self.input()) == 1 and self.output().exists():
                return

            datasets = []
            for tile_type, inp in self.input().items():
                da_embs_tiletype = inp.open()
                if da_embs_tiletype.count() == 0:
                    # scene doesn't contain any tiles
                    continue
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
                    return self.input()[tile_type]

            emb_name = make_embedding_name(
                model_name=self.model_name,
                kind=self.tiles_kind,
                **dict(model_args),
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

    model_name = luigi.Parameter()
    model_args = luigi.DictParameter(default={})

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            tiles_kind=self.tiles_kind,
            model_name=self.model_name,
            model_args=self.model_args,
        )

    def _filter_scene_ids(self, scene_ids):
        if self.tiles_kind == "triplets":
            tiles = self.input().read()
            scene_ids = [scene_id for scene_id in scene_ids if len(tiles[scene_id]) > 0]
        return scene_ids

    def _get_scene_ids_task_class(self):
        if self.tiles_kind == "triplets":
            return TripletSceneSplits
        else:
            return GenerateSceneIDs


def make_transform_name(transform, **transform_args):
    if "pretrained_model" in transform_args:
        return transform_args["pretrained_model"].replace("/", "__")

    if transform is None:
        raise Exception("Provided transform can't be None")

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
    embedding_transform = luigi.OptionalStrParameter()
    embedding_transform_args = luigi.DictParameter(default={})

    def _apply_transform(self, da_emb):
        transform_model_args = dict(self.embedding_transform_args)

        if "pretrained_model" in transform_model_args:
            transform_model_args["pretrained_model"] = joblib.load(
                self.transform_model_filepath
            )

        da_emb, transform_model = apply_transform(
            da=da_emb,
            transform_type=self.embedding_transform,
            return_model=True,
            n_jobs=-1,  # default to using all CPU cores
            **transform_model_args,
        )

        if "pretrained_model" not in self.embedding_transform_args:
            joblib.dump(transform_model, self.transform_model_filepath)

        return da_emb

    @property
    def transform_model_filepath(self):
        fn = f"{self.transform_model_name}.joblib"
        fp_transform_model = Path(self.data_path) / "embeddings" / "models" / fn
        return fp_transform_model

    @property
    def transform_model_name(self):
        transform_model_args = dict(self.embedding_transform_args)

        if "pretrained_model" in transform_model_args:
            return transform_model_args["pretrained_model"]
        else:
            if self.__class__.__module__ == "convml_data.pipeline.embeddings.sampling":
                embedding_model_name = self.model_name
                embedding_model_args = self.model_args
            else:
                embedding_model_name = Path(self.embedding_model_path).name
                embedding_model_args = self.embedding_model_args

            emb_name = make_embedding_name(
                kind=self.tiles_kind,
                model_name=embedding_model_name,
                **embedding_model_args,
            )

            transform_model_name = f"{self.transform_name}.of.{emb_name}"
            return transform_model_name

    @property
    def transform_name(self):
        return make_transform_name(
            transform=self.embedding_transform,
            **dict(self.embedding_transform_args),
        )


class AggregatedDatasetScenesTileEmbeddings(
    luigi.Task,
):
    """
    Combine into a single file all embeddings for all tiles across all scenes in dataset
    """

    tiles_kind = luigi.Parameter()
    data_path = luigi.Parameter(default=".")

    model_name = luigi.Parameter()
    model_args = luigi.DictParameter(default={})

    def requires(self):
        kwargs = dict(
            tiles_kind=self.tiles_kind,
            data_path=self.data_path,
            model_name=self.model_name,
            model_args=self.model_args,
        )

        return DatasetScenesTileEmbeddings(**kwargs)

    def run(self):
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

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        da_all_scenes.to_netcdf(self.output().path)

    def output(self):
        model_args = dict(self.model_args)

        name_parts = ["__all__"]

        if self.tiles_kind == "triplets":
            # remove the tile_type so that it doesn't become part of the
            # directory name
            tile_type = model_args.pop("tile_type", "all")
            if tile_type != "all":
                name_parts.append(tile_type)

        emb_name = make_embedding_name(
            model_name=self.model_name, kind=self.tiles_kind, **dict(model_args)
        )
        path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name

        fn_out = f"{'.'.join(name_parts)}.nc"

        return XArrayTarget(str(path / fn_out))


@luigi.util.requires(AggregatedDatasetScenesTileEmbeddings)
class TransformedAggregatedDatasetScenesTileEmbeddings(
    luigi.Task,
    _AggregatedTileEmbeddingsTransformMixin,
):
    """
    Apply transform across all embeddings from all scenes in dataset
    """

    def run(self):
        da_all_scenes = self.input().open()
        da_all_scenes = self._apply_transform(da_emb=da_all_scenes)
        da_all_scenes.to_netcdf(self.output().path)

    def output(self):
        model_args = dict(self.model_args)

        name_parts = ["__all__"]

        if self.tiles_kind == "triplets":
            # remove the tile_type so that it doesn't become part of the
            # directory name
            tile_type = model_args.pop("tile_type", "all")
            if tile_type != "all":
                name_parts.append(tile_type)

        emb_name = make_embedding_name(
            model_name=self.model_name, kind=self.tiles_kind, **dict(model_args)
        )
        path = Path(self.data_path) / "embeddings" / self.tiles_kind / emb_name

        if self.transform_name is not None:
            name_parts.append(self.transform_name)

        fn_out = f"{'.'.join(name_parts)}.nc"

        return XArrayTarget(str(path / fn_out))
