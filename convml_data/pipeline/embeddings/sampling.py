from pathlib import Path

import luigi
import xarray as xr
from convml_tt.data.dataset import ImageSingletDataset
from convml_tt.data.transforms import get_transforms as get_model_transforms
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings

from ...utils.luigi import XArrayTarget
from ..tiles import SceneTilesData
from .rect.sampling import (  # noqa
    DatasetScenesSlidingWindowImageEmbeddings,
    model_identifier_from_filename,
)


class _SceneTileEmbeddingsBase(luigi.Task):
    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    model_path = luigi.Parameter()
    model_args = luigi.DictParameter(default={})

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

        # TODO: move this
        prediction_batch_size = self.model_args.get("prediction_batch_size", 32)
        da_pred = get_embeddings(
            model=model,
            tile_dataset=tile_dataset,
            prediction_batch_size=prediction_batch_size,
        )
        da_pred["triplet_tile_id"] = tile_dataset.df_tiles.triplet_tile_id
        da_pred.name = "emb"
        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        da_pred.to_netcdf(self.output().path)

    @property
    def tiles_kind(self):
        raise NotImplementedError

    def output(self):
        path = Path(self.data_path) / "embeddings" / self.tiles_kind
        fn_out = f"{self.scene_id}.nc"

        return XArrayTarget(str(path / fn_out))


class SceneTripletTileEmbeddings(_SceneTileEmbeddingsBase):
    """
    Get embeddings for all tiles of a specific type (anchor, neighbor, distant) in a specific scene
    """

    tile_type = luigi.Parameter()
    tiles_kind = "triplets"

    def _get_tile_dataset(self, model_transforms):
        first_tile = next(iter(self.input().values()))
        tiles_path = Path(first_tile["image"].path).parent
        scene_tile_ids = self.input().keys()

        tile_type = self.tile_type

        tile_dataset = ImageSingletDataset(
            data_dir=tiles_path,
            transform=model_transforms,
            tile_type=tile_type.upper(),
            stage=None,
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
        path = Path(self.data_path) / "embeddings" / self.tiles_kind
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
            for tile_type in ["anchor", "neighbor", "distant"]:
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
            path = Path(self.data_path) / "embeddings" / self.tiles_kind
            fn_out = f"{self.scene_id}.nc"

            return XArrayTarget(str(path / fn_out))
