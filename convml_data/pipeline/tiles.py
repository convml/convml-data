from pathlib import Path

import luigi
import regridcart as rc
import xarray as xr

from .. import DataSource
from ..sampling import domain as sampling_domain
from ..sources import create_image as create_source_image
from ..utils.luigi import DBTarget, XArrayTarget, YAMLTarget
from . import trajectory_tiles, triplets
from .aux_sources import CheckForAuxiliaryFiles
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
        p = Path(self.data_path) / self.tiles_kind
        return DBTarget(path=p, db_type="yaml", db_name=f"{self.scene_id}_tiles")


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
                    tiles_meta=tiles_meta,
                    domain=domain,
                    data_source=self.data_source,
                )
            elif self.tiles_kind == "trajectories":
                tile_locations = tiles_meta
            else:
                raise NotImplementedError(self.tiles_kind)
        else:
            tile_locations = []

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tile_locations)

    def output(self):
        name = f"tile_locations.{self.scene_id}"
        p = Path(self.data_path) / self.tiles_kind
        return DBTarget(path=p, db_type="yaml", db_name=name)


class CropSceneSourceFilesForTiles(CropSceneSourceFiles):
    tiles_kind = luigi.Parameter()
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")

    def requires(self):
        reqs = super().requires()

        reqs["tile_locations"] = SceneTileLocations(
            data_path=self.data_path, scene_id=self.scene_id, tiles_kind=self.tiles_kind
        )

        return reqs

    @property
    def domain(self):
        tiles_meta = self.input()["tile_locations"].open()

        lats = []
        lons = []
        for tile_meta in tiles_meta:
            lats.append(tile_meta["loc"]["central_latitude"])
            lons.append(tile_meta["loc"]["central_longitude"])

        da_lat = xr.DataArray(lats)
        da_lon = xr.DataArray(lons)

        domain_spanning = sampling_domain.LatLonPointsSpanningDomain(
            da_lat=da_lat, da_lon=da_lon
        )

        datasource = DataSource.load(path=self.data_path)
        sampling_meta = datasource.sampling
        dx = datasource.sampling["resolution"]
        tile_N = sampling_meta[self.tiles_kind]["tile_N"]
        tile_size = dx * tile_N

        domain = rc.LocalCartesianDomain(
            central_longitude=domain_spanning.central_longitude,
            central_latitude=domain_spanning.central_latitude,
            l_zonal=domain_spanning.l_zonal + 2 * tile_size,
            l_meridional=domain_spanning.l_meridional + 2 * tile_size,
        )

        return domain

    @property
    def output_path(self):
        output_path = super().output_path
        assert output_path.name == "cropped"

        return output_path.parent / f"cropped_for_{self.tiles_kind}"


class SceneTilesData(_SceneRectSampleBase):
    tiles_kind = luigi.Parameter()
    aux_name = luigi.OptionalParameter(default=None)

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
                aux_name=self.aux_name,
            )
        else:
            reqs["source_data"] = CropSceneSourceFilesForTiles(
                scene_id=self.scene_id,
                data_path=self.data_path,
                pad_ptc=self.crop_pad_ptc,
                tiles_kind=self.tiles_kind,
                aux_name=self.aux_name,
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

        if self.aux_name is None:
            source_name = data_source.source
            product = data_source.type
        else:
            source_name = self.data_source.aux_products[self.aux_name]["source"]
            product = self.data_source.aux_products[self.aux_name]["type"]

        tile_N = data_source.sampling[self.tiles_kind].get("tile_N")

        for tile_identifier, tile_domain, tile_meta in self.tile_domains:
            method = "nearest_s2d"
            da_tile = rc.resample(domain=tile_domain, da=da_src, dx=dx, method=method)
            if tile_N is not None:
                tile_shape = (int(da_tile.x.count()), int(da_tile.y.count()))
                if tile_shape[0] != tile_N or tile_shape[1] != tile_N:
                    raise Exception(
                        "Regridder returned a tile with incorrect shape "
                        f"({tile_N}, {tile_N}) != {tile_shape}"
                    )

            if self.aux_name is not None:
                da_tile.name = self.aux_name
                da_tile.attrs.update(da_src.attrs)

            tile_output = self.output()[tile_identifier]
            Path(tile_output["data"].path).parent.mkdir(exist_ok=True, parents=True)
            tile_output["data"].write(da_tile)

            if source_name == "goes16" and product == "truecolor_rgb":
                # to be able to create a RGB image with satpy we need to set the
                # attrs again to ensure we get a proper RGB image
                da_tile.attrs.update(da_src.attrs)

            # if self.aux_name is not None:
            # invert_colors = data_source.aux_products[self.aux_name].get("invert_values_for_rgb", False)
            img_tile = create_source_image(
                da=da_tile, source_name=source_name, product=product
            )
            img_tile.save(str(tile_output["image"].fn))

            tile_meta["scene_id"] = self.scene_id
            if self.aux_name is not None:
                tile_meta["aux_name"] = self.aux_name
            tile_output["meta"].write(tile_meta)

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

            yield tile_identifier, tile_domain, tile_meta

    def get_tile_collection_name(self, tile_meta):
        if self.tiles_kind == "triplets":
            tile_collection_name = tile_meta["triplet_collection"]
        elif self.tiles_kind == "trajectories":
            tile_collection_name = None
        else:
            raise NotImplementedError(self.tiles_kind)

        return tile_collection_name

    def output(self):
        if not self.input()["tile_locations"].exists():
            return luigi.LocalTarget("__fakefile__.nc")

        tiles_meta = self.input()["tile_locations"].open()

        tiles_data_path = Path(self.data_path) / self.tiles_kind
        if self.aux_name is not None:
            tiles_data_path /= self.aux_name

        outputs = {}

        for tile_meta in tiles_meta:
            tile_identifier = self.tile_identifier_format.format(**tile_meta)
            tile_data_path = tiles_data_path

            tile_collection = self.get_tile_collection_name(tile_meta)
            if tile_collection is not None:
                tile_data_path /= tile_collection

            fn_data = f"{tile_identifier}.nc"
            fn_image = f"{tile_identifier}.png"
            fn_meta = f"{tile_identifier}.yml"
            outputs[tile_identifier] = dict(
                data=XArrayTarget(str(tile_data_path / fn_data)),
                image=luigi.LocalTarget(str(tile_data_path / fn_image)),
                meta=YAMLTarget(path=str(tile_data_path / fn_meta)),
            )
        return outputs


class CreateTilesMeta(luigi.Task):
    """
    Generate meta info for all tiles across all scenes. This task is only
    implemented for convenience. To actually generate the tile data for all
    scenes you should use the `GenerateTiles` task
    """

    data_path = luigi.Parameter(default=".")
    tiles_kind = luigi.Parameter()

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        tasks = {}
        if self.tiles_kind == "triplets":
            tasks["tiles_per_scene"] = triplets.TripletSceneSplits(
                data_path=self.data_path
            )
        elif self.tiles_kind == "trajectories":
            tasks["tiles_per_scene"] = trajectory_tiles.TilesPerScene(
                data_path=self.data_path
            )
        else:
            raise NotImplementedError(self.tiles_kind)

        return tasks

    def run(self):
        tiles_per_scene = self.input()["tiles_per_scene"].open()
        if "scene_ids" in self.input():
            scene_ids = list(self.input()["scene_ids"].open().keys())
        else:
            scene_ids = list(tiles_per_scene.keys())

        tasks_tiles = {}
        for scene_id in scene_ids:
            tiles_meta = tiles_per_scene[scene_id]
            if len(tiles_meta) > 0:
                tasks_tiles[scene_id] = SceneTileLocations(
                    data_path=self.data_path,
                    scene_id=scene_id,
                    tiles_kind=self.tiles_kind,
                )

        yield tasks_tiles


class GenerateTiles(luigi.Task):
    """
    Generate all tiles across all scenes. First which tiles to generate per
    scene is worked out (the method is dependent on the sampling strategy of
    the tiles, for example triplets or along trajectories) and second
    `SceneTilesData` is invoked to generate tiles for each scene.
    """

    data_path = luigi.Parameter(default=".")
    tiles_kind = luigi.Parameter()
    aux_name = luigi.OptionalParameter(default=None)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        tasks = {}
        if self.tiles_kind == "triplets":
            tasks["tiles_per_scene"] = triplets.TripletSceneSplits(
                data_path=self.data_path
            )
        elif self.tiles_kind == "trajectories":
            tasks["tiles_per_scene"] = trajectory_tiles.TilesPerScene(
                data_path=self.data_path
            )
        else:
            raise NotImplementedError(self.tiles_kind)

        if self.aux_name is not None:
            tasks["aux_scenes"] = CheckForAuxiliaryFiles(
                data_path=self.data_path, aux_name=self.aux_name
            )

        return tasks

    def run(self):
        tiles_per_scene = self.input()["tiles_per_scene"].open()
        scene_ids = list(tiles_per_scene.keys())

        # exclude scene ids without a tile
        scene_ids = [
            scene_id for scene_id in scene_ids if len(tiles_per_scene[scene_id]) > 0
        ]

        if "aux_scenes" in self.input():
            aux_scene_ids = list(self.input()["aux_scenes"].open().keys())
            scene_ids = [
                scene_id for scene_id in scene_ids if scene_id in aux_scene_ids
            ]

        tasks_tiles = {}
        for scene_id in scene_ids:
            tasks_tiles[scene_id] = SceneTilesData(
                scene_id=scene_id,
                tiles_kind=self.tiles_kind,
                aux_name=self.aux_name,
            )

        yield tasks_tiles
