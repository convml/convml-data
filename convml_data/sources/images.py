"""
Utilities for creating a images representing the scene source data
"""
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..sources import goes16


def save_ax_nosave(ax, **kwargs):
    """
    Create an Image object from the content of a matplotlib.Axes which can be saved to a PNG-file

    source: https://stackoverflow.com/a/43099136/271776
    """
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    buff = io.BytesIO()
    ax.figure.savefig(buff, format="png", dpi=ax.figure.dpi, bbox_inches=bbox, **kwargs)
    ax.axis("on")
    buff.seek(0)
    return Image.open(buff)


def rgb_image_from_scene_data(data_source, product, da_scene):
    if data_source == "goes16":
        if product == "truecolor_rgb" and "bands" in da_scene.coords:
            img_domain = goes16.satpy_rgb.rgb_da_to_img(da=da_scene)
        else:
            height = 5.0
            lx, ly = (
                da_scene.x.max() - da_scene.x.min(),
                da_scene.y.max() - da_scene.y.min(),
            )
            width = height / ly * lx
            fig, ax = plt.subplots(figsize=(width, height))
            ax.set_aspect(1.0)
            da_scene.plot(ax=ax, cmap="nipy_spectral", y="y", add_colorbar=False)
            img_domain = save_ax_nosave(ax=ax)
    else:
        img_data = da_scene.data
        v_min, v_max = np.nanmin(img_data), np.nanmax(img_data)
        img_data = 1.0 - (img_data - v_min) / (v_max - v_min)
        img_data = (img_data * 255).astype(np.uint8)
        img_domain = Image.fromarray(img_data)

    return img_domain
