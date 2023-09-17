from pathlib import Path

import luigi
import regridcart as rc
import xarray as xr

from .. import DataSource
from ..sampling import domain as sampling_domain
from ..utils.luigi import DBTarget, XArrayTarget, YAMLTarget
from . import trajectory_tiles, triplets
from .aux_sources import AuxTaskMixin, CheckForAuxiliaryFiles
from .rect import tiles as rect_tiles
from .regridding import SceneRegriddedData
from .sampling import (
    CropSceneSourceFiles,
    GenerateSceneIDs,
    SceneSourceFiles,
    _SceneRectSampleBase,
)
from .scene_images import SceneImageMixin


class TilesInScene(luigi.Task):
    data_path = luigi.Parameter(default=".")
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()
    extra_args = luigi.DictParameter(default={})

    def requires(self):
        if self.tiles_kind == "triplets":
            return triplets.TripletSceneSplits(data_path=self.data_path)
        if self.tiles_kind == "trajectories":
            return trajectory_tiles.TilesPerScene(data_path=self.data_path)
        if self.tiles_kind == "rect-slidingwindow":
            return rect_tiles.TilesPerRectScene(
                data_path=self.data_path, **dict(self.extra_args)
            )

        raise NotImplementedError(self.tiles_kind)

    def run(self):
        tiles_per_scene = self.input().open()
        # we will write an empty file since we don't need to sample tiles
        # from this scene
        scene_tiles_meta = tiles_per_scene.get(self.scene_id, {})
        if self.tiles_kind == "rect-slidingwindow" and len(scene_tiles_meta) == 0:
            raise Exception
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
    extra_args = luigi.DictParameter(default={})

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        reqs = dict(
            tiles_meta=TilesInScene(
                scene_id=self.scene_id,
                data_path=self.data_path,
                tiles_kind=self.tiles_kind,
                extra_args=self.extra_args,
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
            elif self.tiles_kind == "rect-slidingwindow":
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
    extra_args = luigi.DictParameter(default={})

    def requires(self):
        reqs = super().requires()

        reqs["tile_locations"] = SceneTileLocations(
            data_path=self.data_path,
            scene_id=self.scene_id,
            tiles_kind=self.tiles_kind,
            extra_args=self.extra_args,
        )

        return reqs

    @property
    def domain(self):
        tiles_meta = self.input()["tile_locations"].open()

        if len(tiles_meta) == 0:
            return None

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


class SceneTilesData(_SceneRectSampleBase, SceneImageMixin, AuxTaskMixin):
    """
    Generate all tiles for a specific scene
    """

    tiles_kind = luigi.Parameter()
    aux_name = luigi.OptionalParameter(default=None)
    extra_args = luigi.DictParameter(default={})

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        data_source = self.data_source

        reqs = {}
        if self.tiles_kind == "rect-slidingwindow":
            reqs["source_data"] = SceneRegriddedData(
                data_path=self.data_path,
                scene_id=self.scene_id,
                aux_name=self.aux_name,
            )
        elif self.tiles_kind in ["trajectories", "triplets"]:
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
                    extra_args=self.extra_args,
                )
        else:
            raise NotImplementedError(self.tiles_kind)

        reqs["tile_locations"] = SceneTileLocations(
            data_path=self.data_path,
            scene_id=self.scene_id,
            tiles_kind=self.tiles_kind,
            extra_args=self.extra_args,
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

        tile_N = data_source.sampling[self.tiles_kind].get("tile_N")

        for tile_identifier, tile_domain, tile_meta in self.tile_domains:
            if self.tiles_kind == "rect-slidingwindow":
                da_tile = da_src.isel(**tile_domain)
            else:
                method = "nearest_s2d"
                da_tile = rc.resample(
                    domain=tile_domain, da=da_src, dx=dx, method=method
                )
                if tile_N is not None:
                    img_shape = (int(da_tile.x.count()), int(da_tile.y.count()))
                    if img_shape[0] != tile_N or img_shape[1] != tile_N:
                        raise Exception(
                            "Regridder returned a tile with incorrect shape "
                            f"({tile_N}, {tile_N}) != {img_shape}"
                        )

            if self.aux_name is not None:
                da_tile.name = self.aux_name
            da_tile.attrs.update(da_src.attrs)

            tile_output = self.output()[tile_identifier]
            Path(tile_output["data"].path).parent.mkdir(exist_ok=True, parents=True)
            tile_output["data"].write(da_tile)

            if "image" in tile_output:
                img_tile = self._create_image(da_scene=da_tile)

                if tile_N is not None:
                    if hasattr(img_tile, "size"):
                        img_shape = img_tile.size
                    else:
                        # trollimage.xrimage.XRImage doesn't have a `.size`
                        # attribute like PIL.Image does, but it does have `.data`
                        # which has a shape
                        _, *img_shape = img_tile.data.shape

                    if img_shape[0] != tile_N or img_shape[1] != tile_N:
                        raise Exception(
                            "Produced image has incorrect shape "
                            f"({tile_N}, {tile_N}) != {img_shape}"
                        )
                img_tile.save(str(tile_output["image"].path))

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
        elif self.tiles_kind == "rect-slidingwindow":
            tile_identifier_format = rect_tiles.TILE_IDENTIFIER_FORMAT
        else:
            raise NotImplementedError(self.tiles_kind)

        return tile_identifier_format

    @property
    def tile_domains(self):
        tiles_meta = self.input()["tile_locations"].open()

        for tile_meta in tiles_meta:
            tile_identifier = self.tile_identifier_format.format(**tile_meta)
            if self.tiles_kind == "rect-slidingwindow":
                tile_domain = dict(
                    x=slice(tile_meta["i0"], tile_meta["imax"]),
                    y=slice(tile_meta["j0"], tile_meta["jmax"]),
                )
            else:
                tile_domain = rc.deserialise_domain(tile_meta["loc"])

            yield tile_identifier, tile_domain, tile_meta

    def get_tile_collection_name(self, tile_meta):
        if self.tiles_kind == "triplets":
            tile_collection_name = tile_meta["triplet_collection"]
        elif self.tiles_kind == "trajectories":
            tile_collection_name = None
        elif self.tiles_kind == "rect-slidingwindow":
            tile_collection_name = None
        else:
            raise NotImplementedError(self.tiles_kind)

        return tile_collection_name

    def output(self):
        if not self.input()["tile_locations"].exists():
            return self.input()["tile_locations"]

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
                meta=YAMLTarget(path=str(tile_data_path / fn_meta)),
            )
            if self.image_function is not None:
                outputs[tile_identifier]["image"] = luigi.LocalTarget(
                    str(tile_data_path / fn_image)
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
    extra_args = luigi.DictParameter(default={})

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
        elif self.tiles_kind == "rect-slidingwindow":
            if "step_size" not in self.extra_args:
                raise Exception(
                    "To generate sliding-window tiles on the regridded domain"
                    " `step_size` must be provided in `extra_args`"
                )
            tasks["scene_sources"] = GenerateSceneIDs(data_path=self.data_path)

        else:
            raise NotImplementedError(self.tiles_kind)

        if self.aux_name is not None:
            tasks["aux_scenes"] = CheckForAuxiliaryFiles(
                data_path=self.data_path, aux_name=self.aux_name
            )

        return tasks

    def generate_runtime_tasks(self):
        if self.tiles_kind in ["triplets", "trajectories"]:
            # exclude scene ids without a tile
            tiles_per_scene = self.input()["tiles_per_scene"].open()
            scene_ids = list(tiles_per_scene.keys())

            scene_ids = [
                scene_id for scene_id in scene_ids if len(tiles_per_scene[scene_id]) > 0
            ]
        else:
            scene_ids = self.input()["scene_sources"].open().keys()

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
                extra_args=self.extra_args,
                data_path=self.data_path,
            )

        return tasks_tiles

    def run(self):
        yield self.generate_runtime_tasks()
