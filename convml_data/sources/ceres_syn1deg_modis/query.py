import datetime

import parse
from dateutil.rrule import MONTHLY, rrule

from .api import auth_earthdata_session, find_files_for_month
from .constants import FILENAME_FORMAT, TIME_FORMAT


def make_local_filename(time, version="404406"):
    # I think `401405` and `404406` refer to the versions, they are the two
    # currently available
    fn = FILENAME_FORMAT.format(time=time, version=version)
    return fn


def parse_filename(fn):
    parts = parse.parse(FILENAME_FORMAT, fn).named
    parts["time"] = datetime.datetime.strptime(parts["time"], TIME_FORMAT)
    return parts


def get_available_files(t_start, t_end):
    session = auth_earthdata_session()

    for date_month in rrule(MONTHLY, dtstart=t_start, until=t_end):
        urls = find_files_for_month(date=date_month, session=session)
        for date_url, url in urls.items():
            # every file contains 24 1-hourly samples
            times = [
                datetime.datetime.combine(date_url, datetime.time(hour=i))
                for i in range(24)
            ]
            for time in times:
                if t_start <= time <= t_end:
                    filename = url.split("/")[-1]
                    yield (time, filename)
