"""
Parser and class for representing dataset information given by `meta.yaml` files
"""
import datetime
import functools
import pprint
from pathlib import Path

import dateutil.parser
import isodate
import numpy as np
import xarray as xr
import yaml
from regridcart import LocalCartesianDomain

from .sampling.domain import SourceDataDomain, TrajectoriesSpanningDomain
from .utils import time_filters
from .utils.time import find_nearest_time, npdt64_to_dt


def _parse_datetime(o):
    if not type(o) == datetime.datetime:
        return dateutil.parser.parse(o)
    else:
        return o


def load_trajectories(datasource_meta):
    fn_trajectories = datasource_meta.get("trajectories", {}).get("filepath")
    if fn_trajectories is None:
        raise Exception(
            "Please set the trajectories filepath by defining the value of"
            " `filepath` within `trajectories` in the root of meta.yaml"
        )
    ds_trajectories = xr.open_dataset(fn_trajectories)
    return ds_trajectories


def _parse_time_intervals(time_meta):
    if "intervals" in time_meta:
        for time_interval_meta in time_meta["intervals"]:
            for time_interval in _parse_time_intervals(time_meta=time_interval_meta):
                yield time_interval
    else:
        if "t_start" in time_meta:
            t_start = _parse_datetime(time_meta["t_start"])
            if "N_days" in time_meta:
                duration = datetime.timedelta(days=time_meta["N_days"])
                t_end = t_start + duration
            elif "t_end" in time_meta:
                t_end = _parse_datetime(time_meta["t_end"])
            else:
                raise NotImplementedError(time_meta)
        else:
            raise NotImplementedError(time_meta)

        yield (t_start, t_end)


class DataSource:
    def __init__(self, *args, **kwargs):
        self._meta = kwargs

        self._parse_time_meta()
        self._parse_sampling_meta()
        self._parse_domain_meta()
        self._parse_aux_products()

        assert "source" in self._meta
        assert "type" in self._meta

    def _parse_domain_meta(self):
        domain_meta = self._meta.get("domain", dict(kind="as_source"))
        local_cart_reqd_fields = [
            "central_latitude",
            "central_longitude",
            "l_zonal",
            "l_meridional",
        ]

        if all([field in domain_meta for field in local_cart_reqd_fields]):
            kwargs = {field: domain_meta[field] for field in local_cart_reqd_fields}
            domain = LocalCartesianDomain(**kwargs)
        elif domain_meta.get("kind") == "as_source":
            domain = SourceDataDomain()
        elif domain_meta.get("kind") == "spanning_trajectories":
            ds_trajectories = load_trajectories(datasource_meta=self._meta)
            kwargs = {}
            if "padding" in domain_meta:
                kwargs["padding"] = domain_meta["padding"]
            domain = TrajectoriesSpanningDomain(
                ds_trajectories=ds_trajectories, **kwargs
            )
        else:
            raise NotImplementedError(domain_meta)

        self.domain = domain

    def _parse_sampling_meta(self):
        sampling_meta = self._meta.get("sampling", {})

        if "triplets" or "trajectories" in sampling_meta:
            if "resolution" not in sampling_meta:
                raise Exception(
                    "To do triplet sampling you must define the `resolution` "
                    "(in meters/pixel) in the `sampling` section"
                )

        if "triplets" in sampling_meta:
            required_vars = ["N_triplets"]
            triplets_meta = sampling_meta["triplets"]
            if triplets_meta is None:
                triplets_meta = {}

            if "scene_collections_splitting" not in triplets_meta:
                triplets_meta[
                    "scene_collections_splitting"
                ] = "random_by_relative_sample_size"

            # default tile is 256x256 pixels
            if "tile_N" not in triplets_meta:
                triplets_meta["tile_N"] = 256

            missing_vars = list(filter(lambda v: v not in triplets_meta, required_vars))
            if len(missing_vars) > 0:
                raise Exception(
                    f"To make triplet samplings you must also define the following variables "
                    f" {', '.join(missing_vars)}"
                )

            N_triplets = triplets_meta.get("N_triplets", {})

            # the default triplets collection is called "train"
            if type(N_triplets) == int:
                triplets_meta["N_triplets"] = dict(train=N_triplets)

            assert "train" in triplets_meta["N_triplets"]
            assert "tile_N" in triplets_meta
            # TODO calculate tile size here
            assert "scene_collections_splitting" in triplets_meta
            assert sum(triplets_meta["N_triplets"].values()) > 0

        if "trajectories" in sampling_meta:
            assert "tile_N" in sampling_meta["trajectories"]

        self.sampling = sampling_meta

    def _parse_time_meta(self):
        time_meta = self._meta.get("time")
        if time_meta is None:
            if self.source == "goes16":
                raise Exception(
                    "The goes16 data source requires that you define the start "
                    "time (t_start) and either end time (t_end) or number of days "
                    "(N_days) in a `time` section of `meta.yaml`"
                )
            return
        elif time_meta.get("source") == "trajectories":
            ds_trajectories = load_trajectories(datasource_meta=self._meta)
            da_time = ds_trajectories.time
            t_min = npdt64_to_dt(da_time.min().values)
            t_max = npdt64_to_dt(da_time.max().values)
            self._time_intervals = [(t_min, t_max)]
        else:
            self._time_intervals = list(_parse_time_intervals(time_meta=time_meta))

    def _parse_aux_products(self):
        aux_products_meta = self._meta.get("aux_products", {})
        self.aux_products = {}
        for aux_name, aux_product_meta in aux_products_meta.items():
            assert "source" in aux_product_meta
            assert "type" in aux_product_meta
            if "dt_max" in aux_product_meta:
                aux_product_meta["dt_max"] = np.timedelta64(
                    isodate.parse_duration(aux_product_meta["dt_max"])
                )
            self.aux_products[aux_name] = aux_product_meta

    @property
    def time_intervals(self):
        return self._time_intervals

    @property
    def db_type(self):
        """Use yaml-files as "database" files by default, but make json an option for speed"""
        if "db_type" in self._meta:
            return self._meta["db_type"]
        else:
            return "yaml"

    @classmethod
    @functools.lru_cache(maxsize=10)
    def load(cls, path):
        path_abs = Path(path).expanduser().absolute()
        p = path_abs / "meta.yaml"
        name = p.parent.name
        with open(str(p)) as fh:
            meta = yaml.load(fh.read(), Loader=yaml.FullLoader)
        if meta is None:
            raise Exception(
                "Please as minimum define the `source` and `type` of this "
                "datasource in `meta.yaml`"
            )
        meta["name"] = name
        meta["data_path"] = Path(path_abs).absolute()
        return cls(**meta)

    @property
    def data_path(self):
        return self._meta["data_path"]

    @property
    def source(self):
        return self._meta["source"]

    @property
    def type(self):
        return self._meta["type"]

    @property
    def files(self):
        return self._meta.get("files", None)

    def __repr__(self):
        return pprint.pformat(
            {k: v for k, v in self._meta.items() if not k.startswith("_")}
        )

    def filter_scene_times(self, times):
        """
        Apply the time filtering specified for this source dataset if one is specified
        """
        time_meta = self._meta.get("time")
        if time_meta is None:
            return times

        filters = time_meta.get("filters", {})

        # first we add our own filter which always has to be satisfied: the
        # values fit within the selected time intervals
        def _within_time_intervals(t):
            found_valid_interval = False
            for t_start, t_end in self.time_intervals:
                if np.datetime64(t_start) <= t and t <= np.datetime64(t_end):
                    found_valid_interval = True

            return found_valid_interval

        filter_fns = []
        filter_fns.append(_within_time_intervals)

        map_scene_times_to_trajectory_times = False
        for filter_kind, filter_value in filters.items():
            filter_fn = None
            if filter_kind == "N_hours_from_zenith":
                lon_zenith = self.domain.central_longitude
                filter_fn = functools.partial(
                    time_filters.within_dt_from_zenith,
                    dt_zenith_max=datetime.timedelta(hours=filter_value),
                    lon_zenith=lon_zenith,
                )
            elif filter_kind == time_filters.DATETIME_ATTRS:
                filter_fn = functools.partial(
                    time_filters.within_attr_values, **{filter_kind: filter_value}
                )
            elif filter_kind == "using_trajectory_sampling" and filter_value is True:
                map_scene_times_to_trajectory_times = True
            else:
                raise NotImplementedError(filter_kind)

            if filter_fn is not None:
                filter_fns.append(filter_fn)

        if map_scene_times_to_trajectory_times:
            ds_trajectories = load_trajectories(self._meta)
            da_source_times = ds_trajectories.time
            valid_times = []
            for t in da_source_times.values:
                t_nearest = find_nearest_time(t=t, times=times)
                valid_times.append(t_nearest)
            times = list(set(valid_times))

        for filter_fn in filter_fns:
            times = list(filter(filter_fn, times))

        if len(times) == 0:
            raise Exception("No valid times found")

        return times
