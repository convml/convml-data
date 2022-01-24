import datetime

import parse

URL_FORMAT = (
    "https://asdc.larc.nasa.gov/data/CERES/GEO/Edition4/"
    "{platform_id}_{version}/%Y/{day_of_year:03d}/"
    "CER_GEO_Ed4_{platform_id}_{version}_%Y.{day_of_year:03d}.%H%M.06K.nc"
)


TIME_FORMAT = "%Y%m%d%H%M"
FILENAME_FORMAT = "{time}__{satellite}.nc"


def make_local_filename(time, satellite):
    t_str = time.strftime(TIME_FORMAT)
    fn = FILENAME_FORMAT.format(time=t_str, satellite=satellite)
    return fn


def parse_filename(fn):
    parts = parse.parse(FILENAME_FORMAT, fn).named
    parts["time"] = datetime.datetime.strptime(parts["time"], TIME_FORMAT)
    return parts


def get_available_files(t_start, t_end, satellite):
    # TODO: put in start time for CERES product here
    t0 = datetime.datetime(
        year=t_start.year, month=t_start.month, day=t_start.day, hour=t_start.hour
    )

    if satellite == "goes16n":
        # GOES-16 is every hour half past the hour
        t0 += datetime.timedelta(minutes=30)
    elif satellite == "meteosat9n":
        # Meteosat-9 is every hour on the hour
        pass
    else:
        raise NotImplementedError(satellite)

    if t_start - t0 > datetime.timedelta(minutes=30):
        t0 -= datetime.timedelta(hour=1)

    t = t0
    while t < t_end:
        yield make_local_filename(time=t, satellite=satellite)
        t += datetime.timedelta(hours=1)
