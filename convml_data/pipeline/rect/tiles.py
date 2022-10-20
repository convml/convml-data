from pathlib import Path

import luigi
import numpy as np

try:
    from convml_tt.data.dataset import MovingWindowImageTilingDataset

    HAS_CONVML_TT = True
except ImportError:
    HAS_CONVML_TT = False
from PIL import Image

from ... import DataSource
from ...utils.luigi import DBTarget
from ..regridding import GenerateRegriddedScenes, SceneRegriddedData
from ..utils import SceneBulkProcessingBaseTask

IMAGE_TILE_FILENAME_FORMAT = "{i0:05d}_{j0:05d}.png"
TILE_IDENTIFIER_FORMAT = "{scene_id}__{tile_id:05d}"


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


class TilesPerRectScene(luigi.Task):
    data_path = luigi.Parameter(default=".")
    step_size = luigi.IntParameter()

    def requires(self):
        return GenerateRegriddedScenes(data_path=self.data_path)

    @property
    def datasource(self):
        return DataSource.load(path=self.data_path)

    def run(self):
        inputs = self.input()
        scene_ids = list(inputs.keys())

        sampling_meta = self.datasource.sampling
        sampling_sw = sampling_meta.get("rect-slidingwindow")
        if sampling_sw is None or sampling_sw.get("tile_N") is None:
            raise Exception(
                "Please define the tile-size, in pixels `tile_N` in "
                " a section called `rect-slidingwindow` in `meta.yaml`"
            )
        N_tile = int(sampling_sw["tile_N"])

        tiles_per_scene = {}
        for scene_id in scene_ids:
            scene_tiles = tiles_per_scene.setdefault(str(scene_id), [])

            da_scene = inputs[scene_id]["data"].open()
            nx, ny = int(da_scene.x.count()), int(da_scene.y.count())

            tiler = Tiler(nx=nx, ny=ny, step=self.step_size, N_tile=N_tile)

            for tile_id in np.arange(len(tiler)):
                tiling_slices = tiler[tile_id]

                tile_meta = dict(
                    i0=int(tiling_slices[0].start),
                    imax=int(tiling_slices[0].stop),
                    j0=ny - int(tiling_slices[1].stop),
                    jmax=ny - int(tiling_slices[1].start),
                    scene_id=scene_id,
                    tile_id=int(tile_id),
                )
                scene_tiles.append(tile_meta)

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tiles_per_scene)

    def output(self):
        p = Path(self.data_path) / "rect-slidingwindow"
        return DBTarget(
            path=p, db_type="json", db_name=f"tiles_per_scene.step_{self.step_size}"
        )


class Tiler:
    def __init__(self, nx, ny, step, N_tile):
        self.nx, self.ny = nx, ny

        # assume square tiles and same step in x and y direction
        self.nxt = self.nyt = N_tile
        self.x_step = self.y_step = step

        # the starting image x-index for tile with tile x-index `i`
        self.img_idx_tile_i = np.arange(0, self.nx - self.nxt + 1, self.x_step)
        # the starting image y-index for tile with tile y-index `j`
        self.img_idx_tile_j = np.arange(0, self.ny - self.nyt + 1, self.y_step)

        # number of tiles in x- and y-direction
        self.nt_x = len(self.img_idx_tile_i)
        self.nt_y = len(self.img_idx_tile_j)
        self.num_items = self.nt_x * self.nt_y

    def spatial_index_to_img_ij(self, spatial_index):
        """
        Turn the tile_id (running from 0 to the number of tiles) into the
        (i,j)-index for the tile
        """
        # tile-index in (i,j) running for the number of tiles in each direction
        j = spatial_index // self.nt_x
        i = spatial_index - self.nt_x * j

        # indecies into tile shape
        i_img = self.img_idx_tile_i[i]
        j_img = self.img_idx_tile_j[j]
        return i_img, j_img

    def index_to_spatial_and_scene_index(self, index):
        ntiles_per_scene = self.nt_x * self.nt_y
        scene_index = index // ntiles_per_scene
        spatial_index = index % ntiles_per_scene

        return self.df_rect_images.index[scene_index], spatial_index

    def __getitem__(self, spatial_index):
        i_img, j_img = self.spatial_index_to_img_ij(spatial_index=spatial_index)

        x_slice = slice(i_img, i_img + self.nxt)
        y_slice = slice(j_img, j_img + self.nyt)
        return (x_slice, y_slice)

    def __len__(self):
        return self.num_items
