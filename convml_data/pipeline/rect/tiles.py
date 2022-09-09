from pathlib import Path

import luigi
from convml_tt.data.dataset import MovingWindowImageTilingDataset
from PIL import Image

from .. import SceneBulkProcessingBaseTask, SceneRegriddedData

IMAGE_TILE_FILENAME_FORMAT = "{i0:05d}_{j0:05d}.png"


class SlidingWindowImageTiles(luigi.Task):
    """
    Generate image files for the tiles sampled from a larger rect domain
    """

    image_path = luigi.Parameter()
    src_data_path = luigi.OptionalParameter()
    step_size = luigi.IntParameter(default=10)

    def run(self):
        raise NotImplementedError("changed dataloader")
        img = Image.open(self.image_path)

        N_tile = (256, 256)
        tile_dataset = MovingWindowImageTilingDataset(
            img=img,
            transform=None,
            step=(self.step_size, self.step_size),
            N_tile=N_tile,
        )
        for n in range(len(tile_dataset)):
            tile_img = tile_dataset.get_image(n)
            i, j = tile_dataset.spatial_index_to_img_ij(n)
            filename = IMAGE_TILE_FILENAME_FORMAT.format(i0=i, j0=j)
            tile_filepath = Path(self.output().fn).parent / filename
            tile_img.save(str(tile_filepath))

    def output(self):
        image_fullpath = Path(self.image_path)
        image_path, image_fn = image_fullpath.parent, image_fullpath.name

        tile_path = Path(image_path) / "tiles" / image_fn.replace(".png", "")
        tile_path.mkdir(exist_ok=True, parents=True)

        fn_tile00 = IMAGE_TILE_FILENAME_FORMAT.format(i=0, j=0)
        p = tile_path / fn_tile00
        return luigi.LocalTarget(str(p))


class SceneSlidingWindowImageTiles(SlidingWindowImageTiles):
    """
    Generate image files for the tiles sampled from a larger rect domain for a
    specific scene from a dataset
    """

    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()

    def requires(self):
        return SceneRegriddedData(data_path=self.data_path, scene_id=self.scene_id)

    @property
    def image_path(self):
        return self.input()["image"].fn

    @property
    def src_data_path(self):
        return self.input()["data"].fn

    def output(self):
        dir_name = "{}_tiles_{}step".format(self.scene_id, self.step_size)
        p_out = Path(self.data_path) / "composites" / "rect" / "tiles" / dir_name
        p_out.mkdir(exist_ok=True, parents=True)
        # TODO: fix this, we should actually be working out how many tiles to
        # produce from the source dataset
        fn_tile00 = IMAGE_TILE_FILENAME_FORMAT.format(i0=0, j0=0)
        p = p_out / fn_tile00
        return luigi.LocalTarget(str(p))


class DatasetScenesSlidingWindowImageTiles(SceneBulkProcessingBaseTask):
    step_size = luigi.Parameter()
    TaskClass = SceneSlidingWindowImageTiles

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            step_size=self.step_size,
        )
