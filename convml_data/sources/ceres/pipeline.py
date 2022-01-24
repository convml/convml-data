from pathlib import Path

import luigi

from ...utils.luigi import DBTarget, XArrayTarget
from .download import download_file, make_url
from .query import get_available_files, parse_filename


class QueryForData(luigi.Task):
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
        return parse_filename(fn=filename)["time"]


class FetchFile(luigi.Task):
    data_path = luigi.Parameter()
    filename = luigi.Parameter()

    def run(self):
        file_info = parse_filename(fn=self.filename)
        time = file_info["time"]
        satellite = file_info["satellite"]
        if satellite == "goes16n":
            platform_id = "GOE16_NH"
        elif satellite == "meteosat9n":
            platform_id = "MET09_NH"
        else:
            raise NotImplementedError(satellite)

        version = "V01.2"
        url = make_url(platform_id=platform_id, version=version, time=time)
        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        download_file(url, self.output().path)

    def output(self):
        return XArrayTarget(Path(self.data_path) / self.filename)
