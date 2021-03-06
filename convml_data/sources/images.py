"""
Utilities for creating a images representing the scene source data
"""
import numpy as np
import xarray as xr
from PIL import Image

from ..sources import goes16


def make_rgb(da, alpha=0.5, invert_values=False, **coord_components):
    """
    turn three components along a particular coordinate into RGB values by
    scaling each by its max and min-values in the dataset
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
    elif v_dim not in da.dims:
        raise Exception(f"The `{v_dim}` coordinate wasn't found the provided DataArray")

    if invert_values:

        def scale_zero_one(v):
            return 1.0 - (v - v.min()) / (v.max() - v.min())

    else:

        def scale_zero_one(v):
            return (v - v.min()) / (v.max() - v.min())

    scale = scale_zero_one

    all_dims = da.dims
    x_dim, y_dim = list(filter(lambda d: d != v_dim, all_dims))

    da_rgba = xr.DataArray(
        np.zeros((4, len(da[x_dim]), len(da[y_dim]))),
        dims=("rgba", x_dim, y_dim),
        coords={"rgba": np.arange(4), x_dim: da[x_dim], y_dim: da[y_dim]},
    )

    def _make_component(da_):
        if da_.rgba.data == 3:
            return alpha * np.ones_like(da_)
        else:
            return scale(da.sel({v_dim: dim_idxs[da_.rgba.item()]}).values)

    da_rgba = da_rgba.groupby("rgba").apply(_make_component)

    return da_rgba


def rgb_image_from_scene_data(source_name, product, da_scene, **kwargs):
    if source_name == "goes16":
        if product == "truecolor_rgb" and "bands" in da_scene.coords:
            img_domain = goes16.satpy_rgb.rgb_da_to_img(da=da_scene)
        elif product.startswith("multichannel__") or product.startswith(
            "singlechannel__"
        ):
            if product.startswith("multichannel__"):
                channels = list(goes16.parse_product_shorthand(product).keys())
                # TODO: for now we will invert the Radiance channel values when
                # creating RGB images from them
                da_rgba = make_rgb(
                    da_scene, channel=channels, invert_values=True, alpha=1.0
                )
            elif product.startswith("singlechannel__"):
                raise NotImplementedError(product)
            else:
                raise NotImplementedError(product)

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
