from pathlib import Path

import dateutil.parser
import isodate
import luigi
import satdata

from ...utils.luigi import XArrayTarget, YAMLTarget

DERIVED_PRODUCTS = satdata.Goes16AWS.PRODUCTS.keys()


class DatetimeListParameter(luigi.Parameter):
    def parse(self, x):
        return [dateutil.parser.parse(s) for s in x.split(",")]

    def serialize(self, x):
        return ",".join([t.isoformat() for t in x])


class GOES16Query(luigi.Task):
    dt_max = luigi.FloatParameter()
    channel = luigi.ListParameter()
    time = luigi.DateMinuteParameter()
    debug = luigi.BoolParameter(default=True)
    data_path = luigi.Parameter()
    product = luigi.OptionalParameter(default=None)

    @classmethod
    def get_time(cls, filename):
        return cls.parse_filename(filename=filename)["start_time"]

    @staticmethod
    def parse_filename(filename):
        return satdata.Goes16AWS.parse_key(filename, parse_times=True)

    def run(self):
        cli = satdata.Goes16AWS(offline=False)

        kws = dict()
        if self.channel is None:
            if self.product is None:
                raise Exception("Either channel or produce should be defined")
            kws["product"] = self.product
        else:
            kws["channel"] = self.channel

        filenames = cli.query(
            time=self.time, region="F", debug=self.debug, dt_max=self.dt_max, **kws
        )
        times_and_files = {
            self.parse_filename(fn)["start_time"]: fn for fn in filenames
        }

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        self.output().write(times_and_files)

    def output(self):
        if self.channel is not None:
            filename_format = "ch{channel}_keys_{time}_{duration}.yaml"
        else:
            filename_format = "{product}_keys_{time}_{duration}.yaml"
        fn = filename_format.format(
            channel=self.channel,
            product=self.product,
            time=self.time.isoformat().replace(":", "").replace("-", ""),
            duration=isodate.duration_isoformat(self.dt_max),
        )
        p = Path(self.data_path) / fn
        return YAMLTarget(str(p))


class GOES16Fetch(luigi.Task):
    filename = luigi.Parameter()
    data_path = luigi.Parameter()
    offline_cli = luigi.BoolParameter(default=False)

    @property
    def cli(self):
        local_storage_dir = Path(self.data_path).expanduser()
        return satdata.Goes16AWS(
            offline=self.offline_cli, local_storage_dir=local_storage_dir
        )

    def run(self):
        self.cli.download([self.filename])

    def output(self):
        return XArrayTarget(str(Path(self.data_path) / self.filename))
