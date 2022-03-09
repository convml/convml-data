"""
Utilities for creating a images representing the scene source data
"""
import numpy as np
import xarray as xr
from PIL import Image

from ..sources import goes16


def make_rgb(da, alpha=0.5, invert_values=False, scaling=None, **coord_components):
    """
    turn three components along a particular coordinate into RGB values by
    scaling each by its max and min-values in the dataset. If `scaling` isn't
    provided then the min-max for each component will be used for scaling
    >> make_rgb(da, emb_dim=[0,1,2])
    """
    if len(coord_components) != 1:
        raise Exception(
            "You should provide exactly one coordinate to turn into RGB values"
        )

    v_dim, dim_idxs = list(coord_components.items())[0]

    if len(dim_idxs) != 3:
        raise Exception(
            f"You should provide exactly three indexes of the `{v_dim}` coordinate to turn into RGB."
            f"{coord_components} was supplied"
        )
    if v_dim not in da.dims:
        raise Exception(f"The `{v_dim}` coordinate wasn't found the provided DataArray")

    all_dims = da.dims
    x_dim, y_dim = list(filter(lambda d: d != v_dim, all_dims))

    da_rgba = xr.DataArray(
        np.zeros((4, len(da[x_dim]), len(da[y_dim]))),
        dims=("rgba", x_dim, y_dim),
        coords={"rgba": np.arange(4), x_dim: da[x_dim], y_dim: da[y_dim]},
    )

    def _make_component(da_):
        # the alpha channel is the 4th channel and should just be set to a constant
        if da_.rgba.data == 3:
            return alpha * np.ones_like(da_)

        # otherwise we will extract along the provided coordinate and scale
        # appropriately
        component = dim_idxs[da_.rgba.item()]
        vals = da.sel({v_dim: component}).values

        if scaling is None:
            v_min, v_max = vals.min(), vals.max()
        else:
            v_min, v_max = scaling
        v_range = v_max - v_min

        v_scaled = (vals - v_min) / v_range

        if invert_values:
            v_scaled = 1.0 - v_scaled

        return v_scaled

    da_rgba = da_rgba.groupby("rgba").apply(_make_component)

    return da_rgba


def rgb_image_from_scene_data(source_name, product, da_scene):
    if source_name == "goes16":
        if product == "truecolor_rgb" and "bands" in da_scene.coords:
            img_domain = goes16.satpy_rgb.rgb_da_to_img(da=da_scene)
        elif product.startswith("multichannel__"):
            channels = goes16.parse_product_shorthand(product)

            # check if we're using brightness temperature for all channels
            for _, prefix in channels.items():
                if prefix != "bt":
                    raise NotImplementedError(channels)

            # assuming all channels are brightness temperature we apply the
            # same scaling for all channels, from 260K to 300K
            scaling = (260, 300)

            da_rgba = make_rgb(
                da_scene,
                invert_values=True,
                alpha=1.0,
                scaling=scaling,
                channel=list(channels.keys()),
            )

            # PIL assumes image shape (h, w, channels), indexes from
            # bottom-left of image, no need for alpha values and requires
            # unsigned 8bit ints
            rgb_values = (
                (255.0 * da_rgba.transpose("y", "x", "rgba"))
                .astype(np.uint8)
                .values[::-1, :, :3]
            )
            img_domain = Image.fromarray(rgb_values)
        else:
            raise NotImplementedError(product)
    else:
        img_data = da_scene.data
        v_min, v_max = np.nanmin(img_data), np.nanmax(img_data)
        img_data = 1.0 - (img_data - v_min) / (v_max - v_min)
        img_data = (img_data * 255).astype(np.uint8)
        img_domain = Image.fromarray(img_data)

    return img_domain
