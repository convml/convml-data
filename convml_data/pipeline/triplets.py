import os
from pathlib import Path

import luigi
import numpy as np

from .. import DataSource
from ..sampling import triplets as triplet_sampling
from ..utils.luigi import DBTarget
from . import GenerateSceneIDs

# duplicated from `convml_tt`, do not change this
TILE_IDENTIFIER_FORMAT = "{triplet_id:05d}_{tile_type}"


class TripletSceneSplits(luigi.Task):
    """
    Work out which scenes all triplets should be sampled from
    """

    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)

    @staticmethod
    def _split_scene_ids(scene_ids, method, N_triplets):
        scene_collections = {}
        if method == "random_by_relative_sample_size":
            # split all scene IDs randomly so that for each collection in
            # `N_triplets` the fraction of scenes allocated equals the fraction
            # of triplets in the collection
            N_scenes_total = len(scene_ids)
            N_triplets_total = sum(N_triplets.values())
            scene_ids_shuffled = np.random.permutation(scene_ids)

            def split_list(arr, idx):
                return arr[:idx], arr[idx:]

            for i, (collection_name, N_triplets_collection) in enumerate(
                N_triplets.items()
            ):
                if i <= N_scenes_total - 1:
                    f = N_triplets_collection / N_triplets_total
                    N_scenes_collection = int(f * N_scenes_total)
                else:
                    N_scenes_collection = len(scene_ids_shuffled)

                collection_scene_ids, scene_ids_shuffled = split_list(
                    scene_ids_shuffled, N_scenes_collection
                )
                scene_collections[collection_name] = collection_scene_ids
        else:
            raise NotImplementedError(method)

        return scene_collections

    def run(self):
        scene_ids = list(self.input().open().keys())

        ds = self.data_source

        if "triplets" not in ds.sampling:
            raise Exception(
                "To produce triplets please define a `triplets` section "
                "under `sampling` for the dataset meta info. At minimum "
                "it should contain the number of triplets (`N_triplets` , "
                "this can be a dictionary to have multiple sets, e.g. "
                "'train', 'study', etc), the tile pixel resolution meters (`resolution`) "
                "and the number of pixels in the tile (`tile_N`)"
            )

        triplets_meta = ds.sampling["triplets"]
        N_triplets = triplets_meta["N_triplets"]
        if type(N_triplets) == int:
            N_triplets = dict(train=N_triplets)
        scene_collections_splitting = triplets_meta["scene_collections_splitting"]

        if not len(scene_ids) >= 2:
            raise Exception(
                "At least 2 scenes are needed to do `random_by_relative_sample_size`."
                " Please increase the number of scenes in your data source"
            )

        scene_ids_by_collection = self._split_scene_ids(
            scene_ids=scene_ids,
            method=scene_collections_splitting,
            N_triplets=N_triplets,
        )

        tiles_per_scene = {}
        # first populate the tiles_per_scene collection so there's an entry for
        # each scene ID
        for scene_id in scene_ids:
            tiles_per_scene[scene_id] = []

        for triplet_collection, n_triplets in N_triplets.items():
            collection_scene_ids = scene_ids_by_collection[triplet_collection]
            for n in range(n_triplets):
                # pick two random scene IDs, ensuring that they are different
                scene_id_anchor, scene_id_distant = np.random.choice(
                    collection_scene_ids, size=2, replace=False
                )

                scene_ids = [scene_id_anchor, scene_id_distant]

                for scene_id, is_distant in zip(scene_ids, [False, True]):
                    tiles_per_scene[scene_id].append(
                        dict(
                            triplet_id=n,
                            is_distant=is_distant,
                            triplet_collection=triplet_collection,
                        )
                    )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(tiles_per_scene)

    def output(self):
        p = Path(self.data_path) / "triplets"
        return DBTarget(path=p, db_type="yaml", db_name="tile_scene_splits")


def sample_triplet_tile_locations(tiles_meta, domain, data_source):
    triplets_meta = data_source.sampling["triplets"]
    neigh_dist_scaling = triplets_meta.get("neigh_dist_scaling", 1.0)
    dx = data_source.sampling["resolution"]
    tile_N = triplets_meta["tile_N"]
    tile_size = dx * tile_N

    tile_locations = []
    for n, tile_meta in enumerate(tiles_meta):
        # to ensure that we don't get repeated tile locations generated when
        # running with luigi (which uses multiprocessing) we need to initiate a
        # random-number generator here while passing in the process id
        rng_seed = (
            os.getpid(),
            n,
            tile_meta["triplet_id"],
        )
        rng = np.random.default_rng(rng_seed)

        triplet_tile_locations = triplet_sampling.generate_triplet_location(
            domain=domain,
            tile_size=tile_size,
            neigh_dist_scaling=neigh_dist_scaling,
            rng=rng,
        )

        tile_types = []
        if tile_meta["is_distant"]:
            tile_types.append("distant")
            triplet_tile_locations = triplet_tile_locations[-1:]
        else:
            tile_types.append("anchor")
            tile_types.append("neighbor")
            triplet_tile_locations = triplet_tile_locations[:-1]

        for (tile_type, tile_domain) in zip(tile_types, triplet_tile_locations):
            tile_meta = dict(
                loc=tile_domain.serialize(),
                tile_type=tile_type,
                triplet_id=tile_meta["triplet_id"],
                triplet_collection=tile_meta["triplet_collection"],
            )
            tile_locations.append(tile_meta)

    return tile_locations
