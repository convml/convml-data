"""
luigi Tasks for producing embeddings with a trained neural network across a
whole dataset
"""
from pathlib import Path

import luigi
import xarray as xr
from convml_tt.data.dataset import MovingWindowImageTilingDataset
from convml_tt.data.transforms import get_transforms as get_model_transforms
from convml_tt.system import TripletTrainerModel
from convml_tt.utils import get_embeddings

from .... import DataSource
from ....utils.luigi import XArrayTarget
from ... import SceneBulkProcessingBaseTask, SceneRegriddedData
from ...rect.tiles import DatasetScenesSlidingWindowImageTiles


def model_identifier_from_filename(fn):
    return fn.replace(".torch.pkl", "").replace(".ckpt", "")


class SlidingWindowImageEmbeddings(luigi.Task):
    """
    Make embeddings for a Cartesian 2D image using a specific model skipping
    every `step_size` grid-points in x and y. If `src_data_path` is provided
    then the attributes from that will be copied over (for example lat lon
    coordinates)
    """

    model_path = luigi.Parameter()
    image_path = luigi.Parameter()
    src_data_path = luigi.OptionalParameter()
    step_size = luigi.IntParameter(default=10)
    prediction_batch_size = luigi.IntParameter(default=32)

    def run(self):
        model = TripletTrainerModel.load_from_checkpoint(self.model_path)
        model.freeze()

        N_tile = (256, 256)
        model_transforms = get_model_transforms(
            step="predict", normalize_for_arch=model.base_arch
        )

        tile_dataset = MovingWindowImageTilingDataset(
            data_dir=Path(self.image_path).parent,
            transform=model_transforms,
            step=(self.step_size, self.step_size),
            N_tile=N_tile,
        )

        # remove all but the single image we're interested in
        df_tiles = tile_dataset.df_tiles
        filename_scene = Path(self.image_path).name
        df_tiles_scene = df_tiles[df_tiles.scene_filepath == filename_scene]
        tile_dataset.df_tiles = df_tiles_scene

        if len(tile_dataset) == 0:
            raise Exception("The provided tile-dataset doesn't contain any tiles! ")

        da_pred = get_embeddings(
            model=model,
            tile_dataset=tile_dataset,
            prediction_batch_size=self.prediction_batch_size,
        )

        if self.src_data_path:
            da_src = xr.open_dataarray(self.src_data_path)
            da_pred["x"] = da_src.x.isel(x=da_pred.i0)
            # OBS: j-indexing is positive "down" (from top-left corner) whereas
            # y-indexing is positive "up", so we have to reverse y here before
            # slicing
            Ny = len(da_src.y)
            da_pred["y"] = da_src.y.isel(y=Ny - da_pred.j0)

            da_pred = da_pred.set_index(tile_id=("x", "y")).unstack("tile_id")
            da_pred.attrs["lx_tile"] = float(da_src.x[N_tile[0]] - da_src.x[0])
            da_pred.attrs["ly_tile"] = float(da_src.y[N_tile[1]] - da_src.y[0])

            # make sure the lat lon coords are available later
            da_pred.coords["lat"] = da_src.lat.sel(x=da_pred.x, y=da_pred.y)
            da_pred.coords["lon"] = da_src.lon.sel(x=da_pred.x, y=da_pred.y)

        da_pred.attrs["model_path"] = self.model_path
        da_pred.attrs["image_path"] = self.image_path
        da_pred.attrs["src_data_path"] = self.src_data_path

        da_pred.name = "emb"

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_pred.to_netcdf(self.output().fn)

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        fn_out = image_fn.replace(
            ".png", ".embeddings.{}_step.nc".format(self.step_size)
        )
        return XArrayTarget(str(image_path / fn_out))


class SceneSlidingWindowImageEmbeddings(SlidingWindowImageEmbeddings):
    """
    Create sliding-window tiling embeddings using a specific trained model for
    a specific scene of the dataset in `data_path`
    """

    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return SceneRegriddedData(data_path=self.data_path, scene_id=self.scene_id)

    @property
    def image_path(self):
        return self.input()["image"].path

    @property
    def src_data_path(self):
        return self.input()["data"].path

    def output(self):
        fn = "{}.embeddings.{}_step.nc".format(self.scene_id, self.step_size)
        model_name = model_identifier_from_filename(fn=Path(self.model_path).name)
        p_out = Path(self.data_path) / "embeddings" / "rect" / model_name / fn
        return XArrayTarget(str(p_out))


class DatasetScenesSlidingWindowImageEmbeddings(SceneBulkProcessingBaseTask):
    """
    Create sliding-window tiling embeddings using a specific trained model for
    all scenes in the dataset in `data_path`
    """

    model_path = luigi.Parameter()
    step_size = luigi.Parameter()

    TaskClass = SceneSlidingWindowImageEmbeddings

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            model_path=self.model_path,
            step_size=self.step_size,
        )


class DatasetScenesSlidingWindowImagesAndEmbedddings(luigi.Task):
    data_path = luigi.Parameter()
    model_path = luigi.Parameter()
    step_size = luigi.Parameter()
    generate_tile_images = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = {}
        reqs["data"] = DatasetScenesSlidingWindowImageEmbeddings(
            data_path=self.data_path,
            model_path=self.model_path,
            step_size=self.step_size,
        )
        if self.generate_tile_images:
            reqs["images"] = DatasetScenesSlidingWindowImageTiles(
                data_path=self.data_path,
                step_size=self.step_size,
            )
        return reqs

    @property
    def scene_resolution(self):
        data_source = DataSource.load(path=self.data_path)
        if (
            "resolution" not in data_source.sampling
            or data_source.sampling.get("resolution") is None
        ):
            raise Exception(
                "To produce isometric grid resampling of the source data please "
                "define the grid-spacing by defining `resolution` (in meters/pixel) "
                "in the `sampling` part of the data source meta information"
            )
        return data_source.sampling["resolution"]

    def run(self):
        das = []
        for scene_id, input in self.input()["data"].items():
            da = input.open()
            da["scene_id"] = scene_id

            # turn attributes we want to keep into extra coordinates so that
            # we'll have them later
            for v in ["src_data_path", "image_path"]:
                if v in da.attrs:
                    value = da.attrs.pop(v)
                    da[v] = value
            das.append(da)

        da_all = xr.concat(das, dim="scene_id")
        da_all.name = "emb"
        da_all.attrs["step_size"] = self.step_size
        da_all.attrs["model_path"] = self.model_path

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        da_all.to_netcdf(self.output().path)

    def output(self):
        model_name = Path(self.model_path).name.replace(".pkl", "").replace(".ckpt", "")
        fn = "all_embeddings.{}_step.nc".format(self.step_size)
        p = Path(self.data_path) / "embeddings" / "rect" / model_name / fn
        return XArrayTarget(str(p))
