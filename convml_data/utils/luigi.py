import datetime
import json
from pathlib import Path

import dateutil.parser
import luigi
import numpy as np
import xarray as xr
import yaml
from PIL import Image

try:
    import orjson

    HAS_JSON = True
except ImportError:
    HAS_JSON = False

import logging

logger = logging.getLogger("luigi")

import coloredlogs

coloredlogs.install(level="DEBUG", logger=logger)
# coloredlogs.install()


class XArrayTarget(luigi.LocalTarget):
    def open(self, *args, **kwargs):
        # ds = xr.open_dataset(self.path, engine='h5netcdf', *args, **kwargs)
        try:
            ds = xr.open_dataset(self.path, *args, **kwargs)
        except Exception as ex:
            raise Exception(f"There was an issue opening {self.path}") from ex

        if len(ds.data_vars) == 1:
            name = list(ds.data_vars)[0]
            da = ds[name]
            da.name = name
            return da
        else:
            return ds

    @property
    def fn(self):
        return self.path

    def write(self, ds):
        ds.to_netcdf(self.fn)


class YAMLTarget(luigi.LocalTarget):
    def write(self, obj):
        with super().open("w") as fh:
            yaml.dump(obj, fh)

    def read(self):
        with super().open() as fh:
            return yaml.load(fh, Loader=yaml.FullLoader)

    def open(self):
        return self.read()

    def exists(self):
        return Path(self.path).exists()


def _check_for_json():
    if not HAS_JSON:
        raise Exception("To use json please install the `orjson` package")


class JSONTarget(luigi.LocalTarget):
    def __init__(self, *args, **kwargs):
        _check_for_json()
        super().__init__(*args, **kwargs)

    def write(self, obj):
        with super().open("w") as f:
            return json.dump(obj, f)

    def read(self):
        with super().open("rb") as f:
            return orjson.loads(f.read())

    def open(self):
        return self.read()

    def exists(self):
        return Path(self.path).exists()


class DBTarget(luigi.LocalTarget):
    def __init__(self, path, db_type, db_name, *args, **kwargs):
        if db_type == "json":
            _check_for_json()
            ActualTargetClass = JSONTarget
            ext = "json"
        elif db_type == "yaml":
            ActualTargetClass = YAMLTarget
            ext = "yml"
        else:
            raise NotImplementedError(db_type)

        full_path = Path(path) / f"{db_name}.{ext}"
        super().__init__(str(full_path), *args, **kwargs)
        self._actual_target = ActualTargetClass(str(full_path), *args, **kwargs)

    def read(self):
        return self._actual_target.read()

    def write(self, obj):
        return self._actual_target.write(obj)

    def open(self):
        return self.read()  # noqa

    def exists(self):
        return Path(self.path).exists()


class ImageTarget(luigi.LocalTarget):
    def write(self, img):
        img.save(self.path)

    def read(self):
        return Image.open(self.fn)

    def open(self):
        return self.read()

    def exists(self):
        return Path(self.path).exists()


class NumpyDatetimeParameter(luigi.DateSecondParameter):
    def normalize(self, x):
        if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.datetime64):
            dt64 = x
            # https://stackoverflow.com/a/13704307/271776
            ts = (dt64 - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
            return super().normalize(datetime.utcfromtimestamp(ts))
        else:
            try:
                return super().normalize(x)
            except TypeError:
                return super().normalize(dateutil.parser.parse(x))
