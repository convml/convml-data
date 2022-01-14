from pathlib import Path

import luigi
import regridcart as rc

from .. import DataSource
from ..sampling import domain as sampling_domain
from ..utils.domain_images import rgb_image_from_scene_data
from ..utils.luigi import DBTarget, XArrayTarget
from . import trajectory_tiles, triplets
from .sampling import CropSceneSourceFiles, SceneSourceFiles, _SceneRectSampleBase


class TilesInScene(luigi.Task):
    data_path = luigi.Parameter(default=".")
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    def requires(self):
        if self.tiles_kind == "triplets":
            return triplets.TripletSceneSplits(data_path=self.data_path)
        if self.tiles_kind == "trajectories":
            return trajectory_tiles.TilesPerScene(data_path=self.data_path)

        raise NotImplementedError(self.tiles_kind)

    def run(self):
        tiles_per_scene = self.input().open()
        # we will write an empty file since we don't need to sample tiles
        # from this scene
        scene_tiles_meta = tiles_per_scene.get(self.scene_id, {})
        self.output().write(scene_tiles_meta)

    def output(self):
        p = Path(self.data_path) / "tiles"
        return DBTarget(path=p, db_type="yaml", db_name="f{self.scene_id}_tiles")


class SceneTileLocations(luigi.Task):
    """
    For a given scene work out the sampling locations of all the tiles in it
    """

    data_path = luigi.Parameter(default=".")
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        reqs = dict(
            tiles_meta=TilesInScene(
                scene_id=self.scene_id,
                data_path=self.data_path,
                tiles_kind=self.tiles_kind,
            ),
        )

        domain = self.data_source.domain
        if isinstance(domain, sampling_domain.SourceDataDomain):
            reqs["scene_source_data"] = SceneSourceFiles(
                scene_id=self.scene_id, data_path=self.data_path
            )

        return reqs

    def run(self):
        tiles_meta = self.input()["tiles_meta"].open()

        if len(tiles_meta) > 0:
            # not all scenes have to have tiles in them (if for example we're
            # sampling fewer tiles that we have scenes)
            if self.tiles_kind == "triplets":
                domain = self.data_source.domain
                if isinstance(domain, sampling_domain.SourceDataDomain):
                    ds_scene = self.input()["scene_source_data"].open()
                    domain = domain.generate_from_dataset(ds=ds_scene)
                tile_locations = triplets.sample_triplet_tile_locations(
                    tiles_meta=tiles_meta, domain=domain, data_source=self.data_source
                )
            if self.tiles_kind == "trajectories":
                tile_locations = tiles_meta
            else:
                raise NotImplementedError(self.tiles_kind)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tile_locations)

    def output(self):
        name = f"tile_locations.{self.scene_id}"
        p = Path(self.data_path) / "triplets"
        return DBTarget(path=p, db_type="yaml", db_name=name)


class SceneTilesData(_SceneRectSampleBase):
    tiles_kind = luigi.Parameter()

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        data_source = self.data_source

        reqs = {}
        if isinstance(data_source.domain, sampling_domain.SourceDataDomain):
            reqs["source_data"] = SceneSourceFiles(
                scene_id=self.scene_id,
                data_path=self.data_path,
            )
        else:
            reqs["source_data"] = CropSceneSourceFiles(
                scene_id=self.scene_id,
                data_path=self.data_path,
                pad_ptc=self.crop_pad_ptc,
            )

        reqs["tile_locations"] = SceneTileLocations(
            data_path=self.data_path, scene_id=self.scene_id, tiles_kind=self.tiles_kind
        )

        return reqs

    def run(self):
        inputs = self.input()
        source_data_input = inputs["source_data"]
        # for cropped fields the parent task returns a dictionary so that
        # we can have the rendered image too (if that has been produced)
        if isinstance(source_data_input, dict):
            da_src = source_data_input["data"].open()
        else:
            da_src = source_data_input.open()

        domain = self.data_source.domain
        if isinstance(domain, sampling_domain.SourceDataDomain):
            domain = domain.generate_from_dataset(ds=da_src)

        data_source = self.data_source
        dx = data_source.sampling["resolution"]

        for tile_identifier, tile_domain in self.tile_domains:
            da_tile = rc.resample(domain=tile_domain, da=da_src, dx=dx)
            tile_output = self.output()[tile_identifier]
            tile_output["data"].write(da_tile)

            img_tile = rgb_image_from_scene_data(
                data_source=data_source, da_scene=da_tile, src_attrs=da_src.attrs
            )
            img_tile.save(str(tile_output["image"].fn))

    @property
    def tile_identifier_format(self):
        if self.tiles_kind == "triplets":
            tile_identifier_format = triplets.TILE_IDENTIFIER_FORMAT
        elif self.tiles_kind == "trajectories":
            tile_identifier_format = trajectory_tiles.TILE_IDENTIFIER_FORMAT
        else:
            raise NotImplementedError(self.tiles_kind)

        return tile_identifier_format

    @property
    def tile_domains(self):
        tiles_meta = self.input()["tile_locations"].open()

        for tile_meta in tiles_meta:
            tile_domain = rc.deserialise_domain(tile_meta["loc"])
            tile_identifier = self.tile_identifier_format.format(**tile_meta)

            yield tile_identifier, tile_domain

    def output(self):
        if not self.input()["tile_locations"].exists():
            return luigi.LocalTarget("__fakefile__.nc")

        tiles_meta = self.input()["tile_locations"].open()

        tile_data_path = Path(self.data_path) / "triplets"

        outputs = {}

        for tile_meta in tiles_meta:
            tile_identifier = self.tile_identifier_format.format(**tile_meta)
            fn_data = f"{tile_identifier}.nc"
            fn_image = f"{tile_identifier}.png"
            outputs[tile_identifier] = dict(
                data=XArrayTarget(str(tile_data_path / fn_data)),
                image=luigi.LocalTarget(str(tile_data_path / fn_image)),
            )
        return outputs


class GenerateTiles(luigi.Task):
    """
    Generate all tiles across all scenes. First which tiles to generate per
    scene is worked out (the method is dependent on the sampling strategy of
    the tiles, for example triplets or along trajectories) and second
    `SceneTilesData` is invoked to generate tiles for each scene.
    """

    data_path = luigi.Parameter(default=".")
    tiles_kind = luigi.Parameter()

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        if self.tiles_kind == "triplets":
            return triplets.TripletSceneSplits(data_path=self.data_path)
        if self.tiles_kind == "trajectories":
            return trajectory_tiles.TilesPerScene(data_path=self.data_path)

        raise NotImplementedError(self.tiles_kind)

    def run(self):
        tiles_per_scene = self.input().open()

        tasks_tiles = {}
        for scene_id, tiles_meta in tiles_per_scene.items():
            if len(tiles_meta) > 0:
                tasks_tiles[scene_id] = SceneTilesData(
                    scene_id=scene_id, tiles_kind=self.tiles_kind
                )

        yield tasks_tiles
