import numpy as np
import xarray as xr


def cloud_mask_ch13bt(da_bt_13):
    assert da_bt_13.standard_name == "toa_brightness_temperature"
    assert da_bt_13.units == "K"
    ch_no = da_bt_13.channel

    bt_thresholds = [280, 290]
    da_cloudmask = np.logical_and(
        bt_thresholds[0] <= da_bt_13, da_bt_13 <= bt_thresholds[1]
    )
    da_cloudmask.name = "cloud_mask"
    da_cloudmask.attrs[
        "long_name"
    ] = f"{bt_thresholds[0]}K < Tb(ch{ch_no}) < {bt_thresholds[1]}K cloud-mask"
    da_cloudmask.attrs["units"] = "1"
    return da_cloudmask


def inversion_strength(da_q, da_t, da_lnsp, da_z):
    # XXX: return something more useful
    return da_lnsp


def boundary_layer_windspeed(da_u, da_v, da_p):
    import ipdb

    ipdb.set_trace()
    pass


def normalize(value, lower_limit, upper_limit, clip=True):
    """
    Normalize values between 0 and 1 between a lower and upper limit

    Parameters
    ----------
    value :
        The original value. A single value, vector, or array.
    upper_limit :
        The upper limit.
    lower_limit :
        The lower limit.
    clip : bool
        - True: Clips values between 0 and 1 for RGB.
        - False: Retain the numbers that extends outside 0-1 range.
    Output:
        Values normalized between the upper and lower limit.
    """
    norm = (value - lower_limit) / (upper_limit - lower_limit)
    if clip:
        norm = np.clip(norm, 0, 1)
    return norm


def ir_shallow_clouds_ch11_ch14_ch15_v1(da_bt_11, da_bt_14, da_bt_15):
    da_scene = xr.concat([da_bt_11, da_bt_14, da_bt_15], dim="channel")
    return da_scene


def ir_shallow_clouds_ch11_ch14_ch15_v1__img(da_scene):
    bt_min, bt_max = 270, 300
    clip = True

    def brightness_temperature_to_color(bt):
        return 1.0 - normalize(bt, bt_min, bt_max, clip=clip)

    nx = int(da_scene.x.count())
    ny = int(da_scene.y.count())

    arr_img = np.zeros((ny, nx, 3))
    arr_img[..., 0] = brightness_temperature_to_color(da_scene.sel(channel=14))
    da_ch_11_14_avg = 0.5 * (da_scene.sel(channel=11) + da_scene.sel(channel=14))
    arr_img[..., 1] = brightness_temperature_to_color(da_ch_11_14_avg)
    arr_img[..., 2] = brightness_temperature_to_color(da_scene.sel(channel=15))

    return xr.DataArray(arr_img, dims=("y", "x", "rgb"))
