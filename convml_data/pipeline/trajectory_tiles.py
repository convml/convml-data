from pathlib import Path

import luigi
import regridcart as rc

from .. import DataSource, load_trajectories
from ..sampling.domain import LocalCartesianSquareTileDomain
from ..utils.luigi import DBTarget
from ..utils.time import find_nearest_time, npdt64_to_dt
from . import GenerateSceneIDs, parse_scene_id

TILE_IDENTIFIER_FORMAT = "{scene_id}"


class TilesPerScene(luigi.Task):
    data_path = luigi.Parameter(default=".")

    @property
    def datasource(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)

    def run(self):
        scene_ids = list(self.input().open().keys())

        datasource = self.datasource

        if "trajectories" not in datasource.sampling:
            raise Exception(
                "To produce tiles along trajectories please define a `trajectories` section "
                "under `sampling` for the dataset meta info. At minimum "
                "it should contain the tile pixel resolution meters (`resolution`) "
                "and the number of pixels in the tile (`tile_N`)"
            )

        ds_trajectories = load_trajectories(datasource_meta=datasource._meta)
        da_times_traj = ds_trajectories.time

        sampling_meta = datasource.sampling
        dx = datasource.sampling["resolution"]
        tile_N = sampling_meta["trajectories"]["tile_N"]
        tile_size = dx * tile_N

        tiles_per_scene = {}
        for scene_id in scene_ids:
            _, t_scene = parse_scene_id(scene_id)
            t_traj, time_idx = find_nearest_time(
                t=t_scene, times=da_times_traj, return_index=True
            )
            ds_traj_pt = ds_trajectories.sel(time=t_traj)

            if isinstance(datasource.domain, rc.LocalCartesianDomain):
                tile_domain = LocalCartesianSquareTileDomain(
                    central_latitude=ds_traj_pt.lat.values,
                    central_longitude=ds_traj_pt.lon.values,
                    size=tile_size,
                )
            else:
                raise NotImplementedError(datasource.domain)

            scene_tiles = tiles_per_scene.setdefault(str(scene_id), [])

            tile_meta = dict(
                time_idx=time_idx,
                time=npdt64_to_dt(ds_traj_pt.time.values),
                loc=tile_domain.serialize(),
                scene_id=scene_id,
            )
            scene_tiles.append(tile_meta)

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tiles_per_scene)

    def output(self):
        p = Path(self.data_path) / "trajectories"
        return DBTarget(path=p, db_type="yaml", db_name="tiles_per_scene")
