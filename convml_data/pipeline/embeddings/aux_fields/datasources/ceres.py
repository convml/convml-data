import datetime
from pathlib import Path

import luigi
import requests

from ....scene_sources import parse_scene_id
from .. import utils
from .ceres_sw_flux import calc_reflected_sw_from_albedo, calc_toa_net_flux


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

        return


def _download_file(url, local_filename):
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

    with open(local_filename, "wb") as fd:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fd.write(chunk)


class CeresFile(luigi.Task):
    scene_id = luigi.Parameter()
    satelite = luigi.Parameter()
    datapath = luigi.Parameter()

    VERSION = "V01.2"

    URL_FORMAT = (
        "https://asdc.larc.nasa.gov/data/CERES/GEO/Edition4/"
        "{platform_id}_{version}/%Y/{day_of_year:03d}/"
        "CER_GEO_Ed4_{platform_id}_{version}_%Y.{day_of_year:03d}.%H%M.06K.nc"
    )

    def _find_sat_time_and_id(self):
        t = utils.parse_scene_id(self.scene_id)

        if self.satelite == "goes16n":
            platform_id = "GOE16_NH"
            # GOES-16 is every hour half past the hour
            minute = 30
            hour = t.hour
            if t.minute < 0:
                hour -= 1
        elif self.satelite == "meteosat9n":
            platform_id = "MET09_NH"
            # Meteosat-9 is every hour on the hour
            minute = 0
            hour = t.hour
            if t.minute > 30:
                hour += 1
        else:
            raise NotImplementedError(self.satelite)

        t_hr = datetime.datetime(
            year=t.year,
            month=t.month,
            day=t.day,
            hour=hour,
            minute=minute,
        )

        return t_hr, platform_id

    def run(self):
        t_hr, platform_id = self._find_sat_time_and_id()

        url = t_hr.strftime(
            self.URL_FORMAT.format(
                platform_id=platform_id,
                version=self.VERSION,
                day_of_year=t_hr.timetuple().tm_yday,
            )
        )

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        _download_file(url=url, local_filename=self.output().fn)

    def output(self):
        t_hr, platform_id = self._find_sat_time_and_id()
        t_str = t_hr.isoformat().replace(":", "").replace("+", "")
        fn = f"{t_str}__{self.satelite}.nc"
        p = Path(self.datapath) / "source_data" / "CERES" / fn
        return utils.XArrayTarget(str(p))


class CeresFieldCropped(luigi.Task):
    scene_id = luigi.Parameter()
    var_name = luigi.Parameter()
    bbox = utils.BBoxParameter()
    datapath = luigi.Parameter()

    def requires(self):
        _, satelite = self.var_name.split("__")
        return CeresFile(
            scene_id=self.scene_id, satelite=satelite, datapath=self.datapath
        )

    def run(self):
        var_name, _ = self.var_name.split("__")
        ds = self.input().open().rename(longitude="lon", latitude="lat")

        if var_name == "approximate_toa_net_radiation_flux":
            time = parse_scene_id(self.scene_id)[1]
            da_sw_albedo = ds.broadband_shortwave_albedo
            da_lw_flux = ds.broadband_longwave_flux
            da = calc_toa_net_flux(
                da_sw_albedo=da_sw_albedo, da_lw_flux=da_lw_flux, dt_utc=time
            )
            da.name = "approximate_toa_net_radiation_flux"
        elif var_name == "broadband_shortwave_flux":
            time = parse_scene_id(self.scene_id)[1]
            da_sw_albedo = ds.broadband_shortwave_albedo
            da = calc_reflected_sw_from_albedo(da_sw_albedo=da_sw_albedo, dt_utc=time)
            da.name = "broadband_shortwave_flux"
        else:
            da = ds[var_name]

        def scale_km_to_m(da):
            if da.units == "km":
                attrs = dict(da.attrs)
                da *= 1000.0
                da.attrs.update(attrs)
                da.attrs["units"] = "m"
            return da

        da = scale_km_to_m(da)

        mask = (
            (self.bbox["lon"][0] < da.lon)
            * (self.bbox["lon"][1] > da.lon)
            * (self.bbox["lat"][0] < da.lat)
            * (self.bbox["lat"][1] > da.lat)
        )

        da_cropped = da.where(mask, drop=True)

        Path(self.output().fn).parent.mkdir(exist_ok=True, parents=True)
        da_cropped.to_netcdf(str(self.output().fn))

    def output(self):
        fn = f"{self.scene_id}__ceres_{self.var_name}__cropped.nc"
        p = Path(self.requires().output().path).parent / "cropped" / fn
        return utils.XArrayTarget(str(p))
