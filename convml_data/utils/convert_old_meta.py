#!/usr/bin/env python
# coding: utf-8
"""
utility to convert meta info and create yaml-files from old triplet datasets so
that the tiles can be used with convml-data

NOTE: to use this you will first need to modify `meta.yaml` by hand to make
sure it matches the current format for convml-data
"""


from pathlib import Path

import satdata
import yaml
from tqdm import tqdm

import convml_data.pipeline


def _read_tile_meta(fn):
    with open(fn) as fh:
        return yaml.load(fh, Loader=yaml.FullLoader)


def _find_all_unique_scene_files(triplet_filenames):
    filenames = {}
    for fn in tqdm(triplet_filenames):
        tile_meta = _read_tile_meta(fn)
        target_scene_files = tile_meta["target"]["source_files"]
        distant_scene_files = tile_meta["distant"]["source_files"]
        filenames[target_scene_files[0]] = target_scene_files
        filenames[distant_scene_files[0]] = distant_scene_files

    return list(filenames.values())


def _generate_scene_ids(triplet_filenames):
    scenes_filenames = _find_all_unique_scene_files(triplet_filenames)

    scene_ids = {}
    for scene_filenames in scenes_filenames:
        file_meta = satdata.Goes16AWS.parse_key(scene_filenames[0], parse_times=True)
        scene_id = convml_data.pipeline.make_scene_id(
            source="goes16", t_scene=file_meta["start_time"]
        )
        scene_ids[scene_id] = [
            fn.replace("../../../data/storage/sources/goes16", "noaa-goes16")
            for fn in scene_filenames
        ]
    return scene_ids


def _read_scenesplits(triplet_filenames):
    triplet_scenesplits = {}

    for fn in tqdm(triplet_filenames):
        triplet_meta = _read_tile_meta(fn)
        triplet_id = int(Path(fn).name.split("_")[0])
        triplet_collection = Path(fn).parent.name

        tiles_meta = dict(
            anchor=triplet_meta["target"]["source_files"],
            neighbor=triplet_meta["target"]["source_files"],
            distant=triplet_meta["distant"]["source_files"],
        )

        for kind, scene_filenames in tiles_meta.items():
            file_meta = satdata.Goes16AWS.parse_key(
                scene_filenames[0], parse_times=True
            )
            scene_id = convml_data.pipeline.make_scene_id(
                source="goes16", t_scene=file_meta["start_time"]
            )
            split_meta = dict(
                is_distant=kind == "distant",
                triplet_id=triplet_id,
                triplet_collection=triplet_collection,
            )
            scene_triplets = triplet_scenesplits.setdefault(scene_id, [])
            scene_triplets.append(split_meta)

    return triplet_scenesplits


def _read_tile_locations(triplet_filenames):
    tile_locations = {}

    for fn in tqdm(triplet_filenames):
        triplet_meta = _read_tile_meta(fn)
        triplet_id = int(Path(fn).name.split("_")[0])
        triplet_collection = Path(fn).parent.name

        tiles_meta = dict(
            anchor=(
                triplet_meta["target"]["anchor"],
                triplet_meta["target"]["source_files"],
            ),
            neighbor=(
                triplet_meta["target"]["neighbor"],
                triplet_meta["target"]["source_files"],
            ),
            distant=(
                triplet_meta["distant"]["loc"],
                triplet_meta["distant"]["source_files"],
            ),
        )

        for tile_type, (tile_loc_meta, scene_filenames) in tiles_meta.items():
            file_meta = satdata.Goes16AWS.parse_key(
                scene_filenames[0], parse_times=True
            )
            scene_id = convml_data.pipeline.make_scene_id(
                source="goes16", t_scene=file_meta["start_time"]
            )

            scene_tiles = tile_locations.setdefault(scene_id, [])

            tile_meta = dict(
                loc=dict(
                    central_latitude=tile_loc_meta["lat"],
                    central_longitude=tile_loc_meta["lon"],
                    l_meridional=256000.0,
                    l_zonal=256000.0,
                ),
                tile_type=tile_type,
                triplet_collection=triplet_collection,
                triplet_id=triplet_id,
            )
            scene_tiles.append(tile_meta)

    return tile_locations


def main(source_dataset_path):
    path_train = Path(source_dataset_path) / "train"
    path_study = Path(source_dataset_path) / "study"

    with open("meta.yaml") as fh:
        datasource_meta = yaml.load(fh)

    datasource_source = datasource_meta["source"]
    datasource_type = datasource_meta["type"]

    # Generate scene IDs
    triplet_filenames = list(path_train.glob("*meta*.yaml")) + list(
        path_study.glob("*meta*.yaml")
    )

    path_scene_ids = Path(
        f"source_data/{datasource_source}/{datasource_type}/scene_ids.yml"
    )
    path_scene_ids.parent.mkdir(exist_ok=True, parents=True)

    scene_ids = _generate_scene_ids(triplet_filenames)
    with open(path_scene_ids, "w") as fh:
        yaml.dump(scene_ids, stream=fh, default_flow_style=False)
    print(f"wrote scene ids to {path_scene_ids}")

    # Generate triplet tiles meta info
    path_triplet_scenesplits = Path("triplets/tile_scene_splits.yml")
    path_scene_ids.parent.mkdir(exist_ok=True, parents=True)

    triplet_scenesplits = _read_scenesplits(triplet_filenames)

    # check that there are three tiles (triplets) for each triplet_id
    tile_counts = {}

    for scene_tiles in triplet_scenesplits.values():
        for tile_meta in scene_tiles:
            triplet_id = tile_meta["triplet_id"]
            tile_counts.setdefault(triplet_id, 0)
            tile_counts[triplet_id] += 1

    path_triplet_scenesplits.parent.mkdir(exist_ok=True, parents=True)
    with open(path_triplet_scenesplits, "w") as fh:
        yaml.dump(triplet_scenesplits, stream=fh, default_flow_style=False)
    print(f"wrote scene splits to {path_triplet_scenesplits}")

    # tile locations
    tile_locs_per_scene = _read_tile_locations(triplet_filenames)

    for scene_id, tile_locs in tile_locs_per_scene.items():
        path_scene_tile_locs = Path(f"triplets/tile_locations.{scene_id}.yml")
        path_scene_tile_locs.parent.mkdir(exist_ok=True, parents=True)

        with open(path_scene_tile_locs, "w") as fh:
            yaml.dump(tile_locs, stream=fh, default_flow_style=False)
        print(f"wrote tile locations for {scene_id}")


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset-path", help="source dataset path", default="source-dataset"
    )
    args = argparser.parse_args()
    main(source_dataset_path=args.dataset_path)
