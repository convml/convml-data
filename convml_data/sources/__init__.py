from pathlib import Path

import numpy as np
import xarray as xr

from . import ceres, era5, goes16
from .goes16.pipeline import GOES16Fetch, GOES16Query
from .images import rgb_image_from_scene_data as create_image  # noqa
from .les import FindLESFiles, LESDataFile


def build_query_tasks(
    source_name, source_type, time_intervals, source_data_path, product_meta={}
):
    """return collection of luigi.Task objects that will query a data source"""
    if source_name == "goes16":
        if source_type == "truecolor_rgb":
            channels = [1, 2, 3]
        elif source_type.startswith("multichannel__") or source_type.startswith(
            "singlechannel__"
        ):
            channels = list(goes16.parse_product_shorthand(source_type).keys())
        elif source_type in ["image_user_function", "data_user_function"]:
            channels = get_user_function_source_input_names(product_meta=product_meta)
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

        if len(channels) == 1:
            tasks = tasks[channels[0]]

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
    elif source_name in ["ceres", "era5"]:
        if source_name == "ceres":
            QueryTaskClass = ceres.pipeline.QueryForData
        elif source_name == "era5":
            QueryTaskClass = era5.pipeline.ERA5Query
        else:
            raise NotImplementedError(source_name)

        tasks = []
        for t_start, t_end in time_intervals:
            t = QueryTaskClass(
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
    elif source_name == "era5":
        task = era5.pipeline.ERA5Fetch(
            filename=scene_source_files,
            data_path=Path(source_data_path) / "raw",
        )
    else:
        raise NotImplementedError(source_name)

    return task


class IncompleteSourceDefition(Exception):
    pass


def get_user_function_source_input_names(product_meta):
    if "input" not in product_meta:
        raise IncompleteSourceDefition(
            "To use a `user_function` you will need to define its inputs"
            " by listing each channel needed in a section called `input`"
        )
    source_inputs = product_meta["input"]
    if isinstance(source_inputs, list):
        input_names = source_inputs
    elif isinstance(source_inputs, dict):
        input_names = list(source_inputs.keys())
    else:
        raise NotImplementedError(source_inputs)
    return input_names


def get_time_for_filename(source_name, filename):
    """Return the timestamp for a given source data file"""
    if source_name == "goes16":
        t = GOES16Query.get_time(filename=filename)
    elif source_name == "LES":
        t = FindLESFiles.get_time(filename=filename)
    elif source_name == "ceres":
        t = ceres.pipeline.QueryForData.get_time(filename=filename)
    elif source_name == "era5":
        t = era5.pipeline.ERA5Query.get_time(filename=filename)
    else:
        raise NotImplementedError(source_name)

    return t


def _extract_scene_data_for_user_function(task_input, product_meta, bbox_crop):
    das = []
    product_input = product_meta["input"]
    for task_channel in task_input:
        path = task_channel.path
        file_meta = GOES16Query.parse_filename(filename=path)
        channel_number = file_meta["channel"]

        derived_variable = None
        if isinstance(product_input, dict):
            derived_variable = product_input[channel_number]

        da = goes16.satpy_rgb.load_radiance_channel(
            scene_fn=path,
            channel_number=channel_number,
            derived_variable=derived_variable,
            bbox_crop=bbox_crop,
        )
        da["channel"] = channel_number
        das.append(da)
    # TODO: may need to do some regridding here if we try to stack
    # channels with different spatial resolution
    da = xr.concat(das, dim="channel")

    return da


def extract_variable(task_input, data_source, product, product_meta, bbox_crop=None):
    if data_source == "ceres":
        _, var_name = product.split("__")
        da = ceres.extract_variable(task_input=task_input, var_name=var_name)
    elif data_source == "goes16":
        if product == "truecolor_rgb":
            if not len(task_input) == 3:
                raise Exception(
                    "To create TrueColor RGB images for GOES-16 the first"
                    " three Radiance channels (1, 2, 3) are needed"
                )

            scene_fns = [inp.fn for inp in task_input]
            da = goes16.satpy_rgb.load_rgb_files_and_get_composite_da(
                scene_fns=scene_fns, bbox_crop=bbox_crop
            )
        elif product.startswith("multichannel__"):
            das = []
            channels = goes16.parse_product_shorthand(product)
            for task_channel in task_input:
                path = task_channel.path
                file_meta = GOES16Query.parse_filename(filename=path)
                channel_number = file_meta["channel"]

                derived_variable = None
                if channels[channel_number] == "bt":
                    derived_variable = "brightness_temperature"
                elif channels[channel_number] is not None:
                    raise NotImplementedError(channels[channel_number])

                da = goes16.satpy_rgb.load_radiance_channel(
                    scene_fn=path,
                    channel_number=channel_number,
                    derived_variable=derived_variable,
                )
                da["channel"] = channel_number
                das.append(da)
            # TODO: may need to do some regridding here if we try to stack
            # channels with different spatial resolution
            da = xr.concat(das, dim="channel")
        elif product.startswith("singlechannel__"):
            channels = goes16.parse_product_shorthand(product)
            assert len(channels) == 1
            channel_number = list(channels.keys())[0]
            derived_variable = None
            if channels[channel_number] == "bt":
                derived_variable = "brightness_temperature"
            elif channels[channel_number] is not None:
                raise NotImplementedError(channels[channel_number])

            da = goes16.satpy_rgb.load_radiance_channel(
                scene_fn=task_input[0].path,
                derived_variable=derived_variable,
                channel_number=channel_number,
            )
        elif product in ["image_user_function", "data_user_function"]:
            da = _extract_scene_data_for_user_function(
                task_input=task_input, product_meta=product_meta, bbox_crop=bbox_crop
            )

        else:
            da = goes16.satpy_rgb.load_aux_file(scene_fn=task_input)
    elif data_source == "LES":
        ds_input = task_input.open()
        var_name = product
        da = ds_input[var_name]
    elif data_source == "era5":
        da = task_input.open().rename(dict(longitude="lon", latitude="lat"))
    else:
        raise NotImplementedError(data_source)

    return da
