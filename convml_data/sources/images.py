"""
Utilities for creating a images representing the scene source data
"""

import numpy as np
import xarray as xr
from PIL import Image

from ..sources import goes16
from .user_functions import get_user_function


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


def _generic_rgb_from_scene(da_scene):
    if len(da_scene.squeeze().data.shape) == 2:
        da_rgba = make_rgb(
            da=da_scene.squeeze().expand_dims("foo"), alpha=1.0, foo=[0, 0, 0]
        )
        da_rgb = da_rgba.sel(rgba=[0, 1, 2]).rename(dict(rgba="rgb"))
        # need to ensure that the `rgb` dimension is last
        dims = list(da_rgb.dims)
        dims.remove("rgb")
        dims.append("rgb")
        img_data = da_rgb.transpose(*dims).data[::-1, :, :]
    else:
        img_data = da_scene.data
    v_min, v_max = np.nanmin(img_data), np.nanmax(img_data)
    img_data = 1.0 - (img_data - v_min) / (v_max - v_min)
    img_data = (img_data * 255).astype(np.uint8)
    img_domain = Image.fromarray(img_data, "RGB")
    return img_domain


def _rgb_image_from_user_function_scene_data(product, da_scene, **kwargs):
    image_function = get_user_function(**kwargs)
    da_rgb = image_function(da_scene=da_scene)

    if da_rgb.min() < 0.0:
        raise Exception(
            "The minimum value returned from your scaling function should "
            "be at minimum 0"
        )

    if da_rgb.max() > 1.0:
        raise Exception(
            "The maximum value returned from your scaling function should "
            "be at maximum 1"
        )

    if set(da_rgb.dims) != {"x", "y", "rgb"}:
        raise Exception(
            "The shape of the data-array returned from your scaling function"
            "should have coordinates (x, y, rgb)"
        )

    def zero_one_to_255_uint(da):
        return (da * 255).astype(np.uint8)

    rgb_values = (
        zero_one_to_255_uint(da_rgb).transpose("y", "x", "rgb").values[:, :, :3]
    )
    img_domain = Image.fromarray(rgb_values)
    return img_domain


def rgb_image_from_scene_data(source_name, product, da_scene, **kwargs):
    if product == "image_user_function":
        img_domain = _rgb_image_from_user_function_scene_data(
            product=product, da_scene=da_scene, **kwargs
        )
    elif source_name == "goes16":
        if product == "truecolor_rgb" and "bands" in da_scene.coords:
            da_scene.attrs.update(dict(standard_name="true_color"))
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
        img_domain = _generic_rgb_from_scene(da_scene=da_scene)

    return img_domain
