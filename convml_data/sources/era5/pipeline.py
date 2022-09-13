import datetime
from pathlib import Path

import luigi
import luigi.contrib.ssh

from ...utils.luigi import DBTarget, NumpyDatetimeParameter, XArrayTarget

SURFACE_VARIABLES = ["sst"]


def _make_src_filename(
    t=datetime.datetime(year=2020, month=1, day=30, hour=4, minute=0),
    var="t",
):
    if var in SURFACE_VARIABLES:
        level = "sfc"
    else:
        level = "ml"
    fn = t.strftime(
        "/badc/ecmwf-era5/data/oper/an_{level}/%Y/%m/%d/ecmwf-era5_oper_an_{level}_%Y%m%d%H%M.{var}.nc"
    ).format(var=var, level=level)
    return fn


def get_available_files(*args, **kwargs):
    raise NotImplementedError


class ERA5File(luigi.Task):
    host = "JASMIN"
    var_name = luigi.Parameter()
    time = NumpyDatetimeParameter()

    def output(self):
        filepath = _make_src_filename(t=self.time, var=self.var_name)
        print(filepath)
        if self.host == "JASMIN":
            return luigi.contrib.ssh.RemoteTarget(path=filepath, host="jasmin-sci1")
        else:
            return XArrayTarget(path=filepath)


class ERA5Query(luigi.Task):
    t_start = luigi.DateMinuteParameter()
    t_end = luigi.DateMinuteParameter()
    data_type = luigi.OptionalParameter(default=None)
    data_path = luigi.Parameter()

    DB_NAME_FORMAT = "{data_type}_keys_{t_start:%Y%m%d%H%M}_{t_end:%Y%m%d%H%M}"

    def run(self):
        try:
            satellite, _ = self.data_type.split("__")
        except ValueError as ex:
            raise Exception(
                f"Couldn't parse data-type `{self.data_type}`, please make sure"
                " it has the format `{satellite}__{product_name}`"
            ) from ex
        filenames = list(
            get_available_files(
                t_start=self.t_start, t_end=self.t_end, satellite=satellite
            )
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(filenames)

    def output(self):
        db_name = self.DB_NAME_FORMAT.format(
            data_type=self.data_type,
            t_start=self.t_start,
            t_end=self.t_end,
        )
        return DBTarget(path=self.data_path, db_type="yaml", db_name=db_name)

    @staticmethod
    def get_time(filename):
        raise NotImplementedError
        # return parse_filename(fn=filename)["time"]


class ERA5Fetch(luigi.Task):
    pass
