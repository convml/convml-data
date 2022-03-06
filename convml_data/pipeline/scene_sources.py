import datetime
import itertools
import logging
from collections import OrderedDict
from functools import partial
from pathlib import Path

import luigi

from .. import DataSource
from ..sources import build_query_tasks, get_time_for_filename, goes16
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


def merge_multichannel_sources(files_per_channel, time_fn):
    channel_files_by_timestamp = {}
    N_channels = len(files_per_channel)
    for channel, channel_files in files_per_channel.items():
        for ch_filename in channel_files:
            file_timestamp = time_fn(filename=ch_filename)
            time_group = channel_files_by_timestamp.setdefault(file_timestamp, {})
            time_group[channel] = ch_filename

    scene_filesets = []

    for timestamp in sorted(channel_files_by_timestamp.keys()):
        timestamp_files = channel_files_by_timestamp[timestamp]

        if len(timestamp_files) == N_channels:
            scene_filesets.append(
                [timestamp_files[channel] for channel in files_per_channel.keys()]
            )
        else:
            log.warn(
                f"Only {len(timestamp_files)} were found for timestamp {timestamp}"
                " so this timestamp will be excluded"
            )

    return scene_filesets


def create_scenes_from_multichannel_queries(inputs, source_name, product):
    scenes_by_time = {}
    channels_and_filenames = OrderedDict()
    if product == "truecolor_rgb":
        channel_order = [1, 2, 3]
    elif product.startswith("multichannel__"):
        channel_order = list(goes16.parse_product_shorthand(product).keys())
    else:
        raise NotImplementedError(product)

    opened_inputs = {}
    for input_name, input_parts in inputs.items():
        opened_inputs[input_name] = list(
            itertools.chain(*[input_part.open() for input_part in input_parts])
        )

    for channel in channel_order:
        channels_and_filenames[channel] = opened_inputs[channel]

    time_fn = partial(get_time_for_filename, source_name=source_name)

    scene_sets = merge_multichannel_sources(channels_and_filenames, time_fn=time_fn)

    for scene_filenames in scene_sets:
        t_scene = time_fn(filename=scene_filenames[0])
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
            source_type=data_source.type,
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
            scenes_by_time = create_scenes_from_multichannel_queries(
                inputs=self.input(),
                source_name=self.data_source.source,
                product=self.data_source.type,
            )
        elif type(input) == list:
            # single-channel queries
            scenes_by_time = {}
            for input in self.input():
                filename_per_scene = input.open()
                for scene_filename in filename_per_scene:
                    t_scene = get_time_for_filename(
                        filename=scene_filename, source_name=data_source.source
                    )
                    scenes_by_time[t_scene] = scene_filename
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
        path = Path(self.data_path) / "source_data" / ds.source / ds.type
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
