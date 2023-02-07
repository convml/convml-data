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

    DB_NAME_FORMAT = "keys_{t_start:%Y%m%d%H%M}_{t_end:%Y%m%d%H%M}"

    def run(self):
        filenames = dict(get_available_files(t_start=self.t_start, t_end=self.t_end))

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(filenames)

    def output(self):
        db_name = self.DB_NAME_FORMAT.format(
            t_start=self.t_start,
            t_end=self.t_end,
        )
        return DBTarget(path=self.data_path, db_type=self.db_type, db_name=db_name)

    @staticmethod
    def get_time(filename):
        return parse_filename(fn=filename)["time"]


class FetchSyn1DegFile(luigi.Task):
    data_path = luigi.Parameter()
    filename = luigi.Parameter()

    def run(self):
        file_info = parse_filename(fn=self.filename)
        url = make_url(**file_info)
        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        download_file(url, self.output().path)

    def output(self):
        return XArrayTarget(Path(self.data_path) / self.filename)
