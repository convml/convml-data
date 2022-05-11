"""
Utilities for creating a images representing the scene source data
"""
import importlib
from pathlib import Path

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
    if product == "user_function":
        user_function_context = kwargs.get("context", {})
        datasource_path = user_function_context.get("datasource_path", None)
        product_name = user_function_context.get("product_name", None)
        if datasource_path is None:
            raise NotImplementedError(
                "To use a user-function to generate an image you need to pass"
                " in the `datasource_path` as entry in a dict argument called"
                " `context`"
            )
        if product_name is None:
            raise NotImplementedError(
                "To use a user-function to generate an image you need to pass"
                " in the `product_name` as entry in a dict argument called"
                " `context`"
            )

        module_fpath = Path(datasource_path) / "user_functions.py"
        if not module_fpath.exists():
            raise NotImplementedError(
                f"To use user-function you should create a file called {module_fpath.name}"
                f" in {module_fpath.parent} containing a function called `{product_name}`"
                " and taking `da_scene` as an argument"
            )
        spec = importlib.util.spec_from_file_location("user_function", module_fpath)
        user_function_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_function_module)

        image_function = getattr(user_function_module, product_name, None)
        if image_function is None:
            raise NotImplementedError(
                f"Please define `{product_name}` in the user-function module source"
                f" file `{module_fpath}`"
            )

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
    elif source_name == "goes16":
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
