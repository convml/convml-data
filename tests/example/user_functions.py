import numpy as np


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
