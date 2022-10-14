import datetime
import itertools
import logging
from functools import partial
from pathlib import Path

import luigi

from .. import DataSource
from ..sources import build_query_tasks, get_time_for_filename
from ..utils.luigi import DBTarget

log = logging.getLogger()


SCENE_ID_DATE_FORMAT = "%Y%m%d%H%M"


def load_meta(path):
    pass


def make_scene_id(source, t_scene):
    t_str = t_scene.strftime(SCENE_ID_DATE_FORMAT)
    return f"{source}__{t_str}"


def parse_scene_id(s):
    source, t_str = s.split("__")
    return source, datetime.datetime.strptime(t_str, SCENE_ID_DATE_FORMAT)


def merge_multiinput_sources(files_per_input, time_fn):
    input_files_by_timestamp = {}
    N_inputs = len(files_per_input)
    for input_name, input_files in files_per_input.items():
        for input_filename in input_files:
            file_timestamp = time_fn(filename=input_filename)
            time_group = input_files_by_timestamp.setdefault(file_timestamp, {})
            time_group[input_name] = input_filename

    scene_filesets = []

    for timestamp in sorted(input_files_by_timestamp.keys()):
        timestamp_files = input_files_by_timestamp[timestamp]

        if len(timestamp_files) == N_inputs:
            scene_filesets.append(
                {
                    input_name: timestamp_files[input_name]
                    for input_name in files_per_input.keys()
                }
            )
        else:
            log.warn(
                f"Only {len(timestamp_files)} were found for timestamp {timestamp}"
                " so this timestamp will be excluded"
            )

    return scene_filesets


def create_scenes_from_input_queries(inputs, source_name, product):
    """
    Take a collection of "inputs" (named resources with a collection of files
    for each, e.g. a collection of files for the channels `1`, `2` and `3` for
    RGB composite images) and merge these together into scenes with a single
    timestamp for each scene (returned as as a dictionary of times, with each
    entry being a dictionary of the input-name -> filename for that scene)
    """
    scenes_by_time = {}

    opened_inputs = {}
    for input_name, input_parts in inputs.items():
        # some queries return a list of DBTargets each containing a number of
        # filenames matching the queries, some only return a single DBTarget,
        # so we make into a list here to handle the general case
        if not isinstance(input_parts, list):
            input_parts = [input_parts]

        opened_inputs[input_name] = list(
            itertools.chain(*[input_part.open() for input_part in input_parts])
        )

    time_fn = partial(get_time_for_filename, source_name=source_name)

    scene_sets = merge_multiinput_sources(opened_inputs, time_fn=time_fn)

    # use the first input to work out what time to use for the whole combined
    # product for the scene
    input_name_timing = list(inputs.keys())[0]
    for scene_filenames in scene_sets:
        t_scene = time_fn(filename=scene_filenames[input_name_timing])
        scenes_by_time[t_scene] = scene_filenames

    return scenes_by_time


class GenerateSceneIDs(luigi.Task):
    """
    Construct a "database" (actually a yaml or json-file) of all scene IDs in a
    dataset given the source, type and time-span of the dataset. Database
    contains a list of the scene IDs and the sourcefile(s) per scene.
    """

    data_path = luigi.Parameter(default=".")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        data_source = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / data_source.source

        tasks = build_query_tasks(
            source_name=data_source.source,
            product_name=data_source.product,
            source_data_path=source_data_path,
            time_intervals=data_source.time_intervals,
        )

        return tasks

    def run(self):
        data_source = self.data_source

        input = self.input()
        if type(input) == dict:
            # multi-channel queries, each channel is represented by a key in the
            # dictionary
            scenes_by_time = create_scenes_from_input_queries(
                inputs=self.input(),
                source_name=self.data_source.source,
                product=self.data_source.product,
            )
        else:
            raise NotImplementedError(input)

        valid_scene_times = data_source.filter_scene_times(list(scenes_by_time.keys()))

        scenes = {
            make_scene_id(source=data_source.source, t_scene=t_scene): scenes_by_time[
                t_scene
            ]
            for t_scene in valid_scene_times
        }

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(scenes)

    def output(self):
        ds = self.data_source
        name = "scene_ids"
        path = Path(self.data_path) / "source_data" / ds.source / ds.product
        return DBTarget(path=str(path), db_name=name, db_type=ds.db_type)


class DownloadAllSourceFiles(luigi.Task):
    """
    Download all source files for all scenes in the dataset
    """

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return GenerateSceneIDs(data_path=self.data_path)
