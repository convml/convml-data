from .api import auth_earthdata_session
from .constants import URL_FILE_FORMAT


def download_file(url, local_filename):
    session = auth_earthdata_session()
    response = session.get(url)
    response.raise_for_status()

    with open(local_filename, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)


def make_url(time, version):
    day_of_year = time.timetuple().tm_yday
    return URL_FILE_FORMAT.format(day_of_year=day_of_year, time=time, version=version)
