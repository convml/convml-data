import datetime

import luigi
import xarray as xr

SCENE_ID_DATE_FORMAT = "%Y%m%d%H%M"


def parse_scene_id(s):
    data_source, scene_timestamp = s.split("__")
    return datetime.datetime.strptime(scene_timestamp, SCENE_ID_DATE_FORMAT)


class XArrayTarget(luigi.target.FileSystemTarget):
    fs = luigi.local_target.LocalFileSystem()

    def __init__(self, path, *args, **kwargs):
        super(XArrayTarget, self).__init__(path, *args, **kwargs)
        self.path = path

    def open(self, *args, **kwargs):
        try:
            ds = xr.open_dataset(self.path, *args, **kwargs)
        except Exception as ex:
            print(f"Exception was raised trying to open `{self.path}`")
            raise ex

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


class BBoxParameter(luigi.DictParameter):
    def serialize(self, value):
        for v in ["lat", "lon"]:
            if not (v in value and len(value[v]) == 2):
                raise Exception(value)
        return super(BBoxParameter, self).serialize(value)
