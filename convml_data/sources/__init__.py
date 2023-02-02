from pathlib import Path

import numpy as np
import regridcart as rc
import xarray as xr

from . import ceres_geo, ceres_syn1deg_modis, era5, goes16, user_functions
from .goes16.pipeline import GOES16Fetch, GOES16Query
from .images import rgb_image_from_scene_data as create_image  # noqa
from .les import FindLESFiles, LESDataFile


def build_query_tasks(
    source_name, product_name, time_intervals, source_data_path, product_meta={}
):
    """
    Produce a collection of luigi.Task objects that will query a data source
    for the required inputs for producing a specific product.

    Returns dictionary with each element defining the name of an input and
    Task to query available data for that input
    """
    if source_name == "goes16":
        if product_name == "truecolor_rgb":
            input_names = [1, 2, 3]
        elif product_name == "user_function":
            input_names = get_user_function_source_input_names(
                source_name=source_name, product_meta=product_meta
            )
        elif product_name in goes16.DERIVED_PRODUCTS:
            input_names = [product_name]
        else:
            raise NotImplementedError(product_name)

        tasks = {}
        for t_start, t_end in time_intervals:
            dt_total = t_end - t_start
            t_center = t_start + dt_total / 2.0

            for input_name in input_names:
                channel_number = channel_prefix = None
                product = None
                if isinstance(input_name, str):
                    try:
                        channel_number, channel_prefix = goes16.parse_channel_shorthand(
                            channel_idenfitier=input_name
                        )
                        product = None
                    except ValueError:
                        product = product_name
                elif isinstance(input_name, int):
                    channel_number = input_name
                    product = None
                else:
                    raise NotImplementedError(input_name)

                task = GOES16Query(
                    data_path=source_data_path,
                    time=t_center,
                    dt_max=dt_total / 2.0,
                    channel=channel_number,
                    product=product,
                )
                tasks.setdefault(input_name, []).append(task)

    elif source_name == "LES":
        kind, *variables = product_name.split("__")

        if not kind == "singlechannel":
            raise NotImplementedError(product_name)
        source_variable = variables[0]

        raise NotImplementedError("LES")
        data_source = None
        filename_glob = data_source.files is not None and data_source.files or "*.nc"
        tasks = FindLESFiles(
            data_path=source_data_path,
            source_variable=source_variable,
            filename_glob=filename_glob,
        )
    elif source_name in ["ceres_geo", "era5", "ceres_syn1deg_modis"]:
        if source_name == "ceres_geo":
            # CERES has everything in a single-file, so there's just one "data_type"
            QueryTaskClass = ceres_geo.pipeline.QueryForData
            required_inputs = [product_name]
        elif source_name == "ceres_syn1deg_modis":
            # CERES has everything in a single-file, so there's just one "data_type"
            QueryTaskClass = ceres_syn1deg_modis.pipeline.QueryForData
            required_inputs = [product_name]
        elif source_name == "era5":
            # For ERA5 we might make derived variables using multiple files, so
            # the query might be made up of multiple of input files for a
            # single scene
            QueryTaskClass = era5.pipeline.ERA5Query
            if product_name in era5.SOURCE_VARIABLES:
                required_inputs = [
                    product_name,
                ]
            elif product_name == "user_function":
                required_inputs = get_user_function_source_input_names(
                    source_name=source_name, product_meta=product_meta
                )
            elif product_name in era5.DERIVED_VARIABLES:
                required_inputs = _find_source_variables_set(
                    target_variable=product_name,
                    source_variables=era5.SOURCE_VARIABLES,
                    derived_variables={
                        v: prod[1] for (v, prod) in era5.DERIVED_VARIABLES.items()
                    },
                )
            else:
                available_era5_products = list(era5.SOURCE_VARIABLES) + list(
                    era5.DERIVED_VARIABLES.keys()
                )
                raise NotImplementedError(
                    f"{product_name} not known. Known ERA5 products: {', '.join(available_era5_products)}"
                )
        else:
            raise NotImplementedError(source_name)

        tasks = {}
        for t_start, t_end in time_intervals:
            for data_type in required_inputs:
                task = QueryTaskClass(
                    data_path=source_data_path,
                    t_start=t_start,
                    t_end=t_end,
                    data_type=data_type,
                )
                tasks.setdefault(data_type, []).append(task)

    else:
        raise NotImplementedError(source_name)

    return tasks


def _find_source_variables_set(target_variable, source_variables, derived_variables):
    required_inputs = [target_variable]
    # keep iterating until we've found a set of source variables to start from
    max_resolution_depth = 10
    n_resolution_steps = 0
    while any(var_name in derived_variables for var_name in required_inputs):
        n_resolution_steps += 1
        if n_resolution_steps > max_resolution_depth:
            raise Exception(required_inputs)

        for var_name in required_inputs:
            if var_name in derived_variables:
                replacement_vars = derived_variables[var_name]
                required_inputs.remove(var_name)
                required_inputs = list(set(required_inputs).union(replacement_vars))

    while any(var_name in derived_variables for var_name in required_inputs):
        raise Exception(required_inputs)
    if len(required_inputs) == 0:
        raise Exception(target_variable)

    return required_inputs


def build_fetch_tasks(scene_source_files, source_name, source_data_path):
    """Return collection of luigi.Task's which will fetch any missing files for
    a specific scene"""
    kwargs = {}
    if source_name == "goes16":
        kwargs["data_path"] = source_data_path
        FetchTask = GOES16Fetch
    elif source_name == "LES":
        # assume that these files already exist
        FetchTask = LESDataFile
    elif source_name == "ceres_geo":
        kwargs["data_path"] = Path(source_data_path) / "raw"
        FetchTask = ceres_geo.pipeline.FetchFile
    elif source_name == "ceres_syn1deg_modis":
        kwargs["data_path"] = Path(source_data_path) / "raw"
        FetchTask = ceres_syn1deg_modis.pipeline.FetchSyn1DegFile
    elif source_name == "era5":
        kwargs["data_path"] = Path(source_data_path) / "raw"
        FetchTask = era5.pipeline.ERA5Fetch
    else:
        raise NotImplementedError(source_name)

    if not isinstance(scene_source_files, dict):
        raise Exception(
            "Please delete `scene_ids.yaml` and generate again, the stored scene"
            " IDs don't contain a key for the scene components (i.e. they are in"
            " the old format)"
        )

    tasks = {
        input_name: FetchTask(filename=input_filename, **kwargs)
        for (input_name, input_filename) in scene_source_files.items()
    }

    return tasks


class IncompleteSourceDefition(Exception):
    pass


def get_user_function_source_input_names(source_name, product_meta):
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

    if source_name == "era5":
        if any(var_name in era5.DERIVED_VARIABLES for var_name in input_names):
            raise NotImplementedError(input_names)

    return input_names


def _extract_scene_data_for_user_function(task_input, product_meta, bbox_crop):
    das = []

    if product_meta["source"] == "goes16":
        for input_name, input_task in task_input.items():
            path = input_task.path

            # Make it possible to load brightness temperature instead of radiances
            # TODO: this will be redone once I work out a better yaml format
            # for defining how to combine inputs
            derived_variable = None
            if input_name in product_meta["input"]:
                derived_variable = product_meta["input"][input_name]

            da = goes16.satpy_rgb.load_radiance_channel(
                scene_fn=path,
                channel_number=input_name,
                derived_variable=derived_variable,
                bbox_crop=bbox_crop,
            )
            da["channel"] = input_name
            das.append(da)
    else:
        raise NotImplementedError(product_meta["source"])

    # TODO: may need to do some regridding here if we try to stack
    # channels with different spatial resolution
    da = xr.concat(das, dim="channel")

    return da


def _load_goes16_file(path, var_name, bbox_crop=None, **kwargs):
    if var_name in range(17):
        da = goes16.satpy_rgb.load_radiance_channel(
            scene_fn=path, channel_number=var_name, bbox_crop=bbox_crop, **kwargs
        )
        has_cropped = True
    else:
        da, has_cropped = goes16.satpy_rgb.load_aux_file(
            scene_fn=path, bbox_crop=bbox_crop
        )

    return da, has_cropped


def _extract_goes16_variable(product, task_input, domain=None, product_meta={}):
    bbox_crop = None
    if domain is not None:
        # XXX: below is a hack to use the gridding in satpy, this needs
        # refactoring to clean it up

        bbox_lons = domain.latlon_bounds[..., 0]
        bbox_lats = domain.latlon_bounds[..., 1]
        # WESN
        bbox_crop = [bbox_lons.min(), bbox_lons.max(), bbox_lats.min(), bbox_lats.max()]

    if product == "truecolor_rgb":
        try:
            input_keys = [int(v) for v in task_input.keys()]
        except Exception:
            input_keys = list(task_input.keys())
        if not set(input_keys) == {1, 2, 3}:
            raise Exception(
                "To create TrueColor RGB images for GOES-16 the first"
                " three Radiance channels (1, 2, 3) are needed"
            )

        scene_fns = [inp.path for inp in task_input.values()]

        da = goes16.satpy_rgb.load_rgb_files_and_get_composite_da(
            scene_fns=scene_fns, bbox_crop=bbox_crop
        )
    else:
        try:
            channel_number, channel_prefix = goes16.parse_channel_shorthand(
                channel_idenfitier=product
            )
            derived_variable = None
            if channel_prefix == "bt":
                derived_variable = "brightness_temperature"

            da, has_cropped = _load_goes16_file(
                var_name=channel_number,
                path=task_input.path,
                bbox_crop=bbox_crop,
                derived_variable=derived_variable,
            )
            da["channel"] = channel_number
        except ValueError:
            da, has_cropped = _load_goes16_file(
                path=task_input[product].path,
                bbox_crop=bbox_crop,
                var_name=product,
            )

        if domain is not None and not has_cropped:
            da = rc.crop_field_to_domain(domain=domain, da=da)

    if "lat" not in da.coords or "lon" not in da.coords:
        coords = rc.coords.get_latlon_coords_using_crs(da=da)
        # XXX: this is causing extra computation and didn't used to be
        # needed. I need to work out why satpy or something else isn't
        # including that lat/lon variables when interpolating
        da["lat"] = coords["lat"]
        da["lon"] = coords["lon"]

    return da


def _extract_variable_with_user_function():
    pass


def _load_and_crop_input_variable(task, var_name, data_source, domain):
    if data_source == "era5":
        da = task.open().rename(dict(longitude="lon", latitude="lat"))
        if domain is not None:
            da_cropped = rc.crop_field_to_domain(domain=domain, da=da)
            # retain the grid-mapping if present in the source
            if "grid_mapping" in da_cropped.attrs:
                da_cropped.attrs["grid_mapping"] = da.grid_mapping
                da_cropped[da_cropped.grid_mapping] = da[da.grid_mapping]
            da = da_cropped
    elif data_source == "goes16":
        da = _extract_goes16_variable(task_input=task, product=var_name, domain=domain)
    else:
        raise NotImplementedError(data_source)

    return da


def get_required_extra_fields(data_source, product):
    if data_source == "era5" and product in era5.DERIVED_VARIABLES:
        return era5.DERIVED_VARIABLES[product][1]

    return None


def extract_variable(task_input, data_source, product, product_meta={}, domain=None):
    do_crop = domain is not None

    if product == "user_function":
        das = {}
        for input_name, input_task in task_input.items():
            da_src = _load_and_crop_input_variable(
                task=input_task,
                data_source=data_source,
                var_name=input_name,
                domain=domain,
            )
            das[input_name] = da_src
        user_function = user_functions.get_user_function(**product_meta)
        # prefix each variable passed in with `da_` to indicate that it is a DataArray
        kwargs = {f"da_{v}": da for (v, da) in das.items()}
        da = user_function(**kwargs)
        if "long_name" not in da.attrs:
            raise Exception(
                "You should set the `long_name` attribute when creating the"
                f" `{input_name}` field"
            )
        if "units" not in da.attrs:
            raise Exception(
                "You should set the `units` attribute when creating the"
                f" `{input_name}` field"
            )
        do_crop = False

    elif data_source == "ceres_geo":
        _, var_name = product.split("__")
        # CERES files have all products stored in a single file and so for all
        # products there is just a single input
        da = ceres_geo.extract_variable(
            task_input=task_input[product], var_name=var_name
        )
    elif data_source == "ceres_syn1deg_modis":
        var_name = product
        # CERES files have all products stored in a single file and so for all
        # products there is just a single input
        da = ceres_syn1deg_modis.extract_variable(
            task_input=task_input[product], var_name=var_name
        )
    elif data_source == "goes16":
        do_crop = False
        da = _extract_goes16_variable(
            product=product, task_input=task_input, domain=domain
        )
    elif data_source == "LES":
        ds_input = task_input.open()
        var_name = product
        da = ds_input[var_name]
    elif data_source == "era5":
        if product in era5.SOURCE_VARIABLES:
            da = task_input[product].open()
            da = da.rename(dict(longitude="lon", latitude="lat"))
        elif product in era5.DERIVED_VARIABLES:
            calc_fn, _ = era5.DERIVED_VARIABLES[product]
            kwargs = {}
            for (var_name, inp) in task_input.items():
                dsda = inp.open()
                kind = isinstance(dsda, xr.DataArray) and "da" or "ds"
                kwargs[f"{kind}_{var_name}"] = dsda
            # coord order needs to be correct for plotting
            da = calc_fn(**kwargs).transpose(..., "lat", "lon")
        else:
            raise NotImplementedError(product)
    else:
        raise NotImplementedError(data_source)

    if do_crop:
        da_cropped = rc.crop_field_to_domain(domain=domain, da=da)
        return da_cropped
    else:
        return da
