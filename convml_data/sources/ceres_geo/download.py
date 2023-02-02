import requests

URL_FORMAT = (
    "https://asdc.larc.nasa.gov/data/CERES/GEO/Edition4/"
    "{platform_id}_{version}/{time:%Y}/{day_of_year:03d}/"
    "CER_GEO_Ed4_{platform_id}_{version}_{time:%Y}.{day_of_year:03d}.{time:%H%M}.06K.nc"
)


class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = "urs.earthdata.nasa.gov"

    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if "Authorization" in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (
                (original_parsed.hostname != redirect_parsed.hostname)
                and redirect_parsed.hostname != self.AUTH_HOST
                and original_parsed.hostname != self.AUTH_HOST
            ):

                del headers["Authorization"]


def download_file(url, local_filename):
    cookies = {
        "urs_guid_ops": "4a186713-1998-4629-b910-a4e649402ff0",
        "f8283bd54dd0319d97583374486f6b12": "825fe4d91b32ea629e6561d617b6bc8f",
        "65da01e88267beab20264a95b9fc5d5c": "cabc4d82c858483f71fcd054f4df49ff",
        "distribution": "flNiNYWaS4S7nyVZKhN7jKVlWawVaap0S8N7Cuv6NN4Jo7s1lKrQLBrIAgW8wR07mVMbbNsqWBs9zPH++uRtrZ9XYQ7U08lnQONOWZ2ndQPcief16f9DhjhnHYAU74TB",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:85.0) Gecko/20100101 Firefox/85.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.5",
        "DNT": "1",
        "Connection": "keep-alive",
        "Referer": "https://asdc.larc.nasa.gov/data/CERES/GEO/Edition4/MET09_NH_V01.2/2020/013/",
        "Upgrade-Insecure-Requests": "1",
    }

    response = requests.get(url, headers=headers, cookies=cookies)

    response.raise_for_status()

    with open(local_filename, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)


def make_url(platform_id, version, time):
    return URL_FORMAT.format(
        platform_id=platform_id,
        version=version,
        time=time,
        day_of_year=time.timetuple().tm_yday,
    )
