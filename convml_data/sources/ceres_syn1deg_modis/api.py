import datetime
from http.client import NOT_FOUND, UNAUTHORIZED

import bs4
import parse
import requests

from . import earthaccess_auth
from .constants import FILENAME_FORMAT, TIME_FORMAT, URL_MONTH_LISTING_FORMAT


def auth_earthdata_session(strategy="netrc"):
    auth = earthaccess_auth.Auth().login(strategy=strategy)
    if not auth.authenticated:
        auth = earthaccess_auth.Auth().login(strategy="interactive", persist=True)

    # not sure why, but the session object I get out of the earthaccess package
    # doesn't work on its own, but the token inside of it is fine
    session = requests.Session()
    session.headers = {"Authorization": auth.get_session().headers["Authorization"]}
    return session


def find_page_links(url, session):
    """find links in EarthData index page"""
    response = session.get(url)
    if not response.ok:
        if response.status_code == UNAUTHORIZED:
            raise Exception(
                "Earthdata Login reponded with Unauthorized, "
                "did you enter a valid token?"
            )
        if response.status_code == NOT_FOUND:
            raise Exception(
                "The top level URL does not exist, select a URL within "
                "https://asdc.larc.nasa.gov/data/"
            )
    content = response.content
    soup = bs4.BeautifulSoup(content)
    el_table = soup.find("table", {"id": "indexlist"})
    els_links = el_table.find_all("td", {"class": "indexcolname"})
    rel_links = [el.find("a")["href"] for el in els_links]
    abs_links = [url + rel_link for rel_link in rel_links]
    return abs_links


def find_files_for_month(date, session):
    url_month = URL_MONTH_LISTING_FORMAT.format(time=date)
    file_urls = find_page_links(url=url_month, session=session)
    urls_by_time = {}

    for file_url in file_urls:
        time_str = parse.parse(FILENAME_FORMAT, file_url.split("/")[-1])["time"]
        time = datetime.datetime.strptime(time_str, TIME_FORMAT)
        urls_by_time[time] = file_url

    return urls_by_time
