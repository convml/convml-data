from pathlib import Path

import numpy as np
import xarray as xr

from . import ceres, goes16
from . import images as source_images
from .goes16.pipeline import GOES16Fetch, GOES16Query
from .les import FindLESFiles, LESDataFile


def build_query_tasks(source_name, source_type, time_intervals, source_data_path):
    """return collection of luigi.Task objects that will query a data source"""
    if source_name == "goes16":
        if source_type == "truecolor_rgb":
            channels = [1, 2, 3]
        elif source_type.startswith("multichannel__"):
            _, channels_str = source_type.split("__")
            channels = [int(v) for v in channels_str.split("_")]
        else:
            raise NotImplementedError(source_type)

        tasks = {}
        for t_start, t_end in time_intervals:
            dt_total = t_end - t_start
            t_center = t_start + dt_total / 2.0

            for channel in channels:
                t = GOES16Query(
                    data_path=source_data_path,
                    time=t_center,
                    dt_max=dt_total / 2.0,
                    channel=channel,
                )
                tasks.setdefault(channel, []).append(t)
    elif source_name == "LES":
        kind, *variables = source_type.split("__")

        if not kind == "singlechannel":
            raise NotImplementedError(source_type)
        source_variable = variables[0]

        raise NotImplementedError("LES")
        data_source = None
        filename_glob = data_source.files is not None and data_source.files or "*.nc"
        tasks = FindLESFiles(
            data_path=source_data_path,
            source_variable=source_variable,
            filename_glob=filename_glob,
        )
    elif source_name == "ceres":
        tasks = []
        for t_start, t_end in time_intervals:
            t = ceres.pipeline.QueryForData(
                data_path=source_data_path,
                t_start=t_start,
                t_end=t_end,
                data_type=source_type,
            )
            tasks.append(t)

    else:
        raise NotImplementedError(source_name)

    return tasks


def build_fetch_tasks(scene_source_files, source_name, source_data_path):
    """Return collection of luigi.Task's which will fetch any missing files for
    a specific scene"""
    if source_name == "goes16":
        task = GOES16Fetch(
            keys=np.atleast_1d(scene_source_files).tolist(),
            data_path=source_data_path,
        )
    elif source_name == "LES":
        # assume that these files already exist
        task = LESDataFile(file_path=scene_source_files)
    elif source_name == "ceres":
        task = ceres.pipeline.FetchFile(
            filename=scene_source_files,
            data_path=Path(source_data_path) / "raw",
        )
    else:
        raise NotImplementedError(source_name)

    return task


def get_time_for_filename(source_name, filename):
    """Return the timestamp for a given source data file"""
    if source_name == "goes16":
        t = GOES16Query.get_time(filename=filename)
    elif source_name == "LES":
        t = FindLESFiles.get_time(filename=filename)
    elif source_name == "ceres":
        t = ceres.pipeline.QueryForData.get_time(filename=filename)
    else:
        raise NotImplementedError(source_name)

    return t


def extract_variable(task_input, data_source, product):
    if data_source == "ceres":
        _, var_name = product.split("__")
        ds = task_input.open()
        da = ds[var_name].rename(longitude="lon", latitude="lat")
    elif data_source == "goes16":
        if product == "truecolor_rgb":
            if not len(task_input) == 3:
                raise Exception(
                    "To create TrueColor RGB images for GOES-16 the first"
                    " three Radiance channels (1, 2, 3) are needed"
                )

            scene_fns = [inp.fn for inp in task_input]
            da = goes16.satpy_rgb.load_rgb_files_and_get_composite_da(
                scene_fns=scene_fns
            )
        elif product.startswith("multichannel__"):
            das = []
            for task_channel in task_input:
                path = task_channel.path
                file_meta = GOES16Query.parse_filename(filename=path)
                channel_number = file_meta["channel"]
                da = goes16.satpy_rgb.load_radiance_channel(
                    scene_fn=path, channel_number=channel_number
                )
                da["channel"] = channel_number
                das.append(da)
            # TODO: may need to do some regridding here if we try to stack
            # channels with different spatial resolution
            da = xr.concat(das, dim="channel")
        else:
            da = goes16.satpy_rgb.load_aux_file(scene_fn=task_input)
    elif data_source == "LES":
        ds_input = task_input.open()
        var_name = product
        da = ds_input[var_name]
    else:
        raise NotImplementedError(data_source)

    return da


def create_image(da, data_source, product):
    return source_images.rgb_image_from_scene_data(
        data_source=data_source, da_scene=da, product=product
    )
