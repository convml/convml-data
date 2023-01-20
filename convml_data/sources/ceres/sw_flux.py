"""
Functions to calculate the reflected short-wave radiation from the broadband SW
albedo product from CERES
"""
import datetime

from numpy import arcsin, cos, sin
from scipy.constants import pi


def get_julian_day(date):
    return date.timetuple().tm_yday


def solar_declination(dt_utc):
    phi_r = 23.44  # [deg] Earth-tilt angle
    d_y = 365.0  # [1] days in the year
    d_r = 172  # [1] julian day of summer soltice
    d_julian = get_julian_day(dt_utc)
    return phi_r * cos(2.0 * pi * (d_julian - d_r) / d_y)


def elevation_angle(lat, lon, dt_utc):
    phi = lat * pi / 180.0
    le = lon * pi / 180.0
    delta_s_deg = solar_declination(dt_utc=dt_utc)
    delta_s = delta_s_deg * pi / 180.0
    dt_since_midnight = abs(
        dt_utc - datetime.datetime(year=dt_utc.year, month=dt_utc.month, day=dt_utc.day)
    )
    dt_since_midnight.total_seconds()
    return arcsin(
        sin(phi) * sin(delta_s)
        - cos(phi)
        * cos(delta_s)
        * cos(2.0 * pi * dt_since_midnight.total_seconds() / (24 * 60 * 60) + le)
    )


def calc_toa_incoming_sw_flux(lat, lon, dt_utc):
    S0 = 1361.0  # [W/m^2], solar constant. Stull Practical Meteorology p. 41.
    # incoming SW radiation
    da_sw_rad = S0 * elevation_angle(lat=lat, lon=lon, dt_utc=dt_utc)
    da_sw_rad.attrs["units"] = "W m-2"
    return da_sw_rad


def calc_reflected_sw_from_albedo(da_sw_albedo, dt_utc):
    assert da_sw_albedo.units == "%"
    da_sw_rad = calc_toa_incoming_sw_flux(
        lat=da_sw_albedo.lat, lon=da_sw_albedo.lon, dt_utc=dt_utc
    )

    da_sw_reflected = da_sw_rad * da_sw_albedo / 100.0
    attrs = dict(da_sw_albedo.attrs)
    da_sw_reflected.attrs.update(attrs)
    da_sw_reflected.attrs["units"] = "W m-2"
    da_sw_reflected.attrs["long_name"] = "reflected sw flux"

    return da_sw_reflected


def calc_toa_total_outgoing_flux(da_sw_albedo, da_lw_flux, dt_utc):
    assert da_sw_albedo.units == "%"

    da_sw_rad = calc_toa_incoming_sw_flux(
        lat=da_sw_albedo.lat, lon=da_sw_albedo.lon, dt_utc=dt_utc
    )

    da_sw_reflected = da_sw_rad * da_sw_albedo / 100.0

    da_toa_sw_incoming = da_sw_rad - da_sw_reflected
    da_toa_lw_outgoing = da_lw_flux

    da_toa_net_flux = da_toa_sw_incoming - da_toa_lw_outgoing
    da_toa_net_flux.attrs["units"] = "W m-2"
    da_toa_net_flux.attrs["long_name"] = "approximate toa total outgoing radiation flux"
    da_toa_net_flux.name = "approximate_toa_total_outgoing_radiation_flux"
    return da_toa_net_flux
