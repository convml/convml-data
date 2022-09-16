from pathlib import Path

import luigi
import xarray as xr
from convml_tt.data.dataset import ImageSingletDataset
from convml_tt.data.transforms import get_transforms as get_model_transforms
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings

from ...utils.luigi import XArrayTarget
from ..tiles import SceneTilesData
from .defaults import PREDICTION_BATCH_SIZE
from .rect.sampling import (  # noqa
    SLIDING_WINDOW_EMBEDDINGS_DEFAULT_KWARGS,
    DatasetScenesSlidingWindowImageEmbeddings,
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
            raise Exception("The provided tile-dataset doesn't contain any tiles! ")

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
        scene_tile_ids = self.input().keys()

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
            return DatasetScenesSlidingWindowImageEmbeddings(
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
                da_embs_tiletype = inp.open().swap_dims(tile_id="triplet_tile_id")
                da_embs_tiletype["tile_type"] = tile_type
                datasets.append(da_embs_tiletype)

            da = xr.concat(datasets, dim="triplet_tile_id")
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
