import datetime
from pathlib import Path

import luigi
import luigi.contrib.ssh
import parse

from ...utils.luigi import DBTarget, NumpyDatetimeParameter, XArrayTarget
from .base import SOURCE_VARIABLES

SURFACE_VARIABLES = ["sst"]
JASMIN_ROOT_PATH = Path("/badc/ecmwf-era5/data/oper")
TIME_FORMAT = "%Y%m%d%H%M"
FILENAME_FORMAT = "ecmwf-era5_oper_an_{level}_{time}.{var}.nc"


def _make_filepath(
    t=datetime.datetime(year=2020, month=1, day=30, hour=4, minute=0),
    var="t",
):
    """
    Return the filepath for a single era5 file that matches the folder structure on JASMIN
    """
    if var in SURFACE_VARIABLES:
        level = "sfc"
    else:
        level = "ml"

    filename = FILENAME_FORMAT.replace("{time}", "{time:" + TIME_FORMAT + "}").format(
        level=level, time=t, var=var
    )
    data_path = Path(
        "an_{level}/{time:%Y}/{time:%m}/{time:%d}/".format(level=level, time=t)
    )
    return data_path / filename


def parse_filename(fn):
    parts = parse.parse(FILENAME_FORMAT, fn).named
    parts["time"] = datetime.datetime.strptime(parts["time"], TIME_FORMAT)
    return parts


def get_available_files(t_start, t_end, product):
    # era5 is available every hour for all products
    t0 = datetime.datetime(
        year=t_start.year, month=t_start.month, day=t_start.day, hour=t_start.hour
    )

    t = t0
    while t < t_end:
        yield str(_make_filepath(t=t, var=product))
        t += datetime.timedelta(hours=1)


class ERA5File(luigi.Task):
    var_name = luigi.Parameter()
    time = NumpyDatetimeParameter()
    data_root = luigi.Parameter(default=".")

    def output(self):
        filepath = JASMIN_ROOT_PATH / _make_filepath(t=self.time, var=self.var_name)
        return luigi.contrib.ssh.RemoteTarget(path=filepath, host="jasmin-sci1")


class ERA5Query(luigi.Task):
    t_start = luigi.DateMinuteParameter()
    t_end = luigi.DateMinuteParameter()
    data_type = luigi.OptionalParameter(default=None)
    data_path = luigi.Parameter()

    DB_NAME_FORMAT = "{data_type}_keys_{t_start:%Y%m%d%H%M}_{t_end:%Y%m%d%H%M}"

    def run(self):
        if self.data_type not in SOURCE_VARIABLES:
            raise Exception(
                f"{self.data_type} is not one of the available source variables"
            )
        filenames = list(
            get_available_files(
                t_start=self.t_start, t_end=self.t_end, product=self.data_type
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
        return parse_filename(fn=Path(filename).name)["time"]


class ERA5Fetch(luigi.Task):
    data_path = luigi.Parameter()
    filename = luigi.Parameter()

    def requires(self):
        file_info = parse_filename(Path(self.filename).name)
        return ERA5File(
            var_name=file_info["var"], time=file_info["time"], data_root=self.data_path
        )

    def run(self):
        self.input().get(self.output().path)

    def output(self):
        file_info = parse_filename(Path(self.filename).name)
        return XArrayTarget(
            Path(self.data_path)
            / _make_filepath(t=file_info["time"], var=file_info["var"])
        )
