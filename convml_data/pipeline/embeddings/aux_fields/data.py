"""
Produce co-located values of a) cloud organising embeddings and b) a
auxiliary scalar field. There are a few ways this co-location is done:

1. Individual tiles onto which the auxiliary field is regridded and the
   auxiliary is then reduced with a specific operation (for example computing
   the mean)
2. `rect` regridded domains on which embeddings are produced at a fixed spacing
   by sampling tiles on this domain. This produces embedding vectors at
   individual points. These embeddings are then interpolated onto the grid of
   the auxiliary fields.
"""
import importlib
import inspect
from pathlib import Path

import luigi
import luigi.util
import numpy as np
import xarray as xr

from ...aux_sources import CheckForAuxiliaryFiles
from ...embeddings.sampling import (
    SceneTileEmbeddings,
    _AggregatedTileEmbeddingsTransformMixin,
    make_embedding_name,
    make_transform_name,
)
from ...tiles import SceneTilesData
from ...utils import SceneBulkProcessingBaseTask
from . import utils


class SceneAuxFieldWithEmbeddings(luigi.Task):
    aux_name = luigi.OptionalParameter()
    scene_id = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    embedding_model_name = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    data_path = luigi.Parameter(default=".")
    crop_pad_ptc = luigi.FloatParameter(default=0.1)
    tile_reduction_op = luigi.Parameter(default="mean")

    def requires(self):
        tasks = {}
        aux_kwargs = {}
        if self.tiles_kind == "rect-slidingwindow":
            aux_kwargs["step_size"] = self.embedding_model_args["step_size"]

        tasks["aux"] = SceneTilesData(
            data_path=self.data_path,
            aux_name=self.aux_name,
            scene_id=self.scene_id,
            tiles_kind=self.tiles_kind,
            extra_args=aux_kwargs,
        )
        tasks["embeddings"] = SceneTileEmbeddings(
            data_path=self.data_path,
            scene_id=self.scene_id,
            tiles_kind=self.tiles_kind,
            model_name=self.embedding_model_name,
            model_args=self.embedding_model_args,
        )
        return tasks

    def _get_reduction_op(self):
        var_name = f"{self.aux_name}__{self.tile_reduction_op}"
        if hasattr(np, self.tile_reduction_op):
            op_fn = getattr(np, self.tile_reduction_op)

            def reduction_op(da_tile):
                v_long_name = f"tile {self.tile_reduction_op} {da_tile.long_name}"
                da_tile_reduced = op_fn(da_tile)
                da_tile_reduced.attrs["long_name"] = v_long_name
                da_tile_reduced.attrs["units"] = da_tile.units
                da_tile_reduced.name = var_name
                return da_tile_reduced

            return reduction_op

        elif str(self.tile_reduction_op).endswith("__percentile"):
            q = int(str(self.tile_reduction_op).replace("__percentile", ""))

            def reduction_op(da_tile):
                try:
                    suffix = ["st", "nd", "rd"][31 % 10 - 1]
                except IndexError:
                    suffix = "th"

                long_name = f"{q}{suffix} tile percentile of {da_tile.long_name}"
                value = np.percentile(da_tile, q=q)
                units = da_tile.units
                da_tile_reduced = xr.DataArray(
                    value, attrs=dict(long_name=long_name, units=units), name=var_name
                )
                return da_tile_reduced

            return reduction_op

        elif "__" in str(self.tile_reduction_op):
            # e.g. `cloud_metrics__mask__iorg_objects` becomes `cloudmetrics.mask.iorg_objects(...)`
            op_parts = self.tile_reduction_op.split("__")

            fillna_with_zeros = False
            if op_parts[0] == "nanzerofill":
                fillna_with_zeros = True
                op_parts = op_parts[1:]

            module_name = ".".join(op_parts[:-1])
            function_name = op_parts[-1]
            module = importlib.import_module(module_name)
            op_fn = getattr(module, function_name)
            op_desc = " ".join(op_parts[1:])

            argspec = inspect.getargspec(op_fn)
            kwargs = dict()
            if "periodic_domain" in argspec.args:
                # assume that tiles don't represent data on a periodic domain
                # since they've been sampled from a larger domain
                kwargs["periodic_domain"] = False

            def reduction_op(da_tile):
                tile_values = da_tile.values
                if fillna_with_zeros:
                    tile_values = np.nan_to_num(tile_values, nan=0.0)

                agg_value = op_fn(tile_values, **kwargs)
                long_name = f"{op_desc} on {da_tile.long_name}"
                if fillna_with_zeros:
                    long_name += " (nan-zero-filled)"
                # I think most metrics are dimensionless
                units = "1"
                da_tile_reduced = xr.DataArray(
                    agg_value,
                    attrs=dict(long_name=long_name, units=units),
                    name=var_name,
                )
                return da_tile_reduced

            return reduction_op
        else:
            raise NotImplementedError(self.tile_reduction_op)

    def run(self):
        aux_fields = self.input()["aux"]
        da_embs = self.input()["embeddings"].open()

        if da_embs.count() == 0:
            ds = xr.Dataset()
        else:
            if self.tiles_kind == "triplets":
                concat_dim = "triplet_tile_id"
            else:
                concat_dim = "tile_id"
                da_embs = da_embs.stack(n=("x", "y")).swap_dims(n="tile_id").drop("n")

            values = []

            reduction_op = self._get_reduction_op()

            for t_id in da_embs[concat_dim].values:
                da_tile_aux = aux_fields[t_id]["data"].open()
                if "units" not in da_tile_aux.attrs:
                    da_tile_aux.close()
                    Path(aux_fields[t_id]["data"].path).unlink()
                    continue

                da_tile_aux_value = reduction_op(da_tile_aux)
                assert "long_name" in da_tile_aux_value.attrs
                assert "units" in da_tile_aux_value.attrs
                da_tile_aux_value[concat_dim] = t_id
                values.append(da_tile_aux_value)

            if len(values) == 0:
                raise Exception(42)

            da_aux_values = xr.concat(values, dim=concat_dim)
            da_embs = da_embs.rename(t="t_embs")

            ds = xr.merge([da_embs, da_aux_values])

        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        ds.to_netcdf(self.output().path)

    def output(self):
        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_name=self.embedding_model_name,
            **dict(self.embedding_model_args),
        )

        name_parts = [
            self.scene_id,
            self.aux_name,
            f"tile_{self.tile_reduction_op}",
        ]

        fn = ".".join(name_parts) + ".nc"

        p = (
            Path(self.data_path)
            / "embeddings"
            / "with_aux"
            / self.tiles_kind
            / emb_name
            / fn
        )
        return utils.XArrayTarget(str(p))


class DatasetScenesAuxFieldWithEmbeddings(SceneBulkProcessingBaseTask):
    TaskClass = SceneAuxFieldWithEmbeddings
    SceneIDsTaskClass = CheckForAuxiliaryFiles

    aux_name = luigi.OptionalParameter()
    tiles_kind = luigi.Parameter()

    embedding_model_name = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    data_path = luigi.Parameter(default=".")
    crop_pad_ptc = luigi.FloatParameter(default=0.1)
    tile_reduction_op = luigi.Parameter(default="foobar")

    def _get_task_class_kwargs(self, scene_ids):
        return dict(
            aux_name=self.aux_name,
            tiles_kind=self.tiles_kind,
            embedding_model_name=self.embedding_model_name,
            embedding_model_args=self.embedding_model_args,
            tile_reduction_op=self.tile_reduction_op,
        )

    def _get_scene_ids_task_kwargs(self):
        return dict(aux_name=self.aux_name)


class AggregatedDatasetScenesAuxFieldWithEmbeddings(luigi.Task):
    """
    Combine across all scenes in a dataset the embeddings and aux-field values
    into a single file.
    """

    aux_name = luigi.Parameter()
    tiles_kind = luigi.Parameter()

    embedding_model_name = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})

    data_path = luigi.Parameter(default=".")
    tile_reduction_op = luigi.Parameter(default="mean")

    def requires(self):
        kwargs = dict(
            data_path=self.data_path,
            aux_name=self.aux_name,
            tiles_kind=self.tiles_kind,
            embedding_model_name=self.embedding_model_name,
            embedding_model_args=self.embedding_model_args,
            tile_reduction_op=self.tile_reduction_op,
        )

        return DatasetScenesAuxFieldWithEmbeddings(**kwargs)

    def run(self):
        emb_input = self.input()

        datasets = []
        if self.tiles_kind in ["trajectories", "rect-slidingwindow"]:
            concat_dim = "tile_id"
        elif self.tiles_kind == "triplets":
            concat_dim = "triplet_tile_id"
        else:
            raise NotImplementedError(self.tiles_kind)

        for scene_id, inp in emb_input.items():
            ds_scene = inp.open()
            if len(ds_scene.data_vars) == 0:
                # no tiles in this scene
                continue

            ds_scene["scene_id"] = scene_id
            datasets.append(ds_scene)

        import ipdb

        with ipdb.launch_ipdb_on_exception():
            ds = xr.concat(datasets, dim=concat_dim)

        if "stage" in ds.data_vars:
            ds = ds.set_coords(["stage", "scene_id"])
        else:
            ds = ds.set_coords(["scene_id"])

        ds.to_netcdf(self.output().path)

    @property
    def name_parts(self):
        name_parts = [
            "__all__",
            self.aux_name,
            f"tile_{self.tile_reduction_op}",
            "with",
        ]

        return name_parts

    def output(self):
        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_name=self.embedding_model_name,
            **dict(self.embedding_model_args),
        )

        name = ".".join(self.name_parts)
        fn = f"{name}.nc"
        p = (
            Path(self.data_path)
            / "embeddings"
            / "with_aux"
            / self.tiles_kind
            / emb_name
            / fn
        )
        return utils.XArrayTarget(str(p))


@luigi.util.requires(AggregatedDatasetScenesAuxFieldWithEmbeddings)
class TransformedAggregatedDatasetScenesAuxFieldWithEmbeddings(
    luigi.Task, _AggregatedTileEmbeddingsTransformMixin
):
    """
    Apply embedding transform to embeddings that have been interpolated onto
    the grid of an aux variable across entire dataset
    """

    def run(self):
        ds = self.input().open()
        ds["emb"] = self._apply_transform(da_emb=ds.emb)
        Path(self.output().path).parent.mkdir(exist_ok=True, parents=True)
        ds.to_netcdf(self.output().path)

    @property
    def transform_name(self):
        return make_transform_name(
            transform=self.embedding_transform,
            aux=self.aux_name,
            **dict(self.embedding_transform_args),
        )

    @property
    def name_parts(self):
        return [
            "__all__",
            self.aux_name,
            f"tile_{self.tile_reduction_op}",
        ]

    def output(self):
        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_name=self.embedding_model_name,
            **dict(self.embedding_model_args),
        )

        name = ".".join(self.name_parts)
        fn = f"{name}.nc"
        p = (
            Path(self.data_path)
            / "embeddings"
            / "with_aux"
            / self.tiles_kind
            / emb_name
            / f"transformed.with.{self.transform_name}"
            / fn
        )
        return utils.XArrayTarget(str(p))
