"""
No-longer-used routines for regridding embeddings onto the grid of auxiliary
variables. Didn't work so well because the resoluting number of embedding
vectors was enormous
"""
from pathlib import Path

import luigi
import regridcart as rc
import xarray as xr
import xesmf as xe

from ...embeddings.rect.sampling import SceneSlidingWindowImageEmbeddings
from ...embeddings.sampling import make_embedding_name
from ...sampling import CropSceneSourceFiles
from . import data_filters, utils


def scale_km_to_m(da_c):
    assert da_c.units == "km"
    da_c = 1000 * da_c
    da_c.attrs["units"] = "m"
    return da_c


def regrid_emb_var_onto_variable(da_emb, da_v, method="nearest_s2d"):
    """
    Using lat,lon in `da_v` as target grid interpolate `da_emb` on this
    grid using lat,lon in `da_emb` as source grid
    """
    ds_grid_dst = da_v
    # ds_grid_dst = xr.Dataset()
    # ds_grid_dst["lat"] = da_v.lat  # .drop_vars(["lat", "lon"])
    # ds_grid_dst["lon"] = da_v.lon  # .drop_vars(["lat", "lon"])
    # ds_grid_dst

    ds_grid_src = xr.Dataset()
    ds_grid_src["lat"] = da_emb.lat.drop_vars(["lat", "lon"])
    ds_grid_src["lon"] = da_emb.lon.drop_vars(["lat", "lon"])

    regridder = xe.Regridder(
        ds_grid_src,
        ds_grid_dst,
        method,
    )

    if "emb_dim" in da_emb.dims:
        da_emb_regridded = da_emb.groupby("emb_dim").apply(regridder)
    else:
        da_emb_regridded = regridder(da_emb)

    da_emb_regridded.name = da_emb.name
    return da_emb_regridded


class SceneSlidingWindowEmbeddingsAuxFieldRegridding(luigi.Task):
    """
    Regrid embedding variable onto grid of auxillariary field for a single
    scene and optionally reduce using a given operation per segment
    """

    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    aux_name = luigi.Parameter()

    def _parse_filters(self):
        if self.filters is not None:
            return data_filters.parse_defs(self.filters)
        return []

    def requires(self):
        tasks = {}
        tasks["emb"] = SceneSlidingWindowImageEmbeddings(
            data_path=self.data_path,
            scene_id=self.scene_id,
            model_path=self.embedding_model_path,
            **dict(self.embedding_model_args),
        )

        # aux field tasks below
        tasks["aux"] = CropSceneSourceFiles(
            scene_id=self.scene_id,
            data_path=self.data_path,
            aux_name=self.aux_name,
        )

        return tasks

    def reduce(self, da):
        try:
            return getattr(da, self.reduce_op)()
        except AttributeError:
            raise NotImplementedError(self.reduce_op)

    def run(self):
        da_scalar = self.input()["aux"]["data"].open()
        da_emb_src_scene = self.input()["emb"].open()

        # regrid embedding position onto scalar xy-grid (lat,lon)
        if "lat" not in da_scalar.coords:
            # and "lat" not in da_scalar.data_vars:
            coords = rc.coords.get_latlon_coords_using_crs(da=da_scalar)
            da_scalar["lat"] = coords["lat"]
            da_scalar["lon"] = coords["lon"]

        # the xy-coordinate values should come from the embeddings, so let's
        # get rid of them if they're on the scalar data
        for d in ["x", "y"]:
            if d in da_scalar:
                da_scalar = da_scalar.drop(d)

        da_emb = regrid_emb_var_onto_variable(da_emb=da_emb_src_scene, da_v=da_scalar)

        # ensure that the scalar has the name that we requested data for
        da_scalar.name = self.aux_name

        ds = xr.merge([da_scalar, da_emb])

        Path(self.output().fn).parent.mkdir(exist_ok=True)
        ds.to_netcdf(self.output().fn)

    def output(self):
        emb_name = make_embedding_name(
            kind="rect-slidingwindow",
            model_path=self.embedding_model_path,
            **self.embedding_model_args,
        )

        name_parts = [
            self.scene_id,
            self.aux_name,
            "by",
            emb_name,
        ]

        fn = ".".join(name_parts) + ".nc"

        p = Path(self.data_path) / "embeddings" / "with_aux" / fn
        return utils.XArrayTarget(str(p))
