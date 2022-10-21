from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from . import plot_types
from .data import (
    AggregatedDatasetScenesAuxFieldWithEmbeddings,
    make_embedding_name,
    make_transform_name,
)


class AggPlotBaseTask(luigi.Task):
    aux_name = luigi.Parameter()
    scene_ids = luigi.OptionalParameter(default=[])
    tiles_kind = luigi.Parameter()

    data_path = luigi.Parameter(default=".")

    embedding_model_path = luigi.Parameter()
    embedding_model_args = luigi.DictParameter(default={})
    embedding_transform = luigi.Parameter(default=None)
    embedding_transform_args = luigi.DictParameter(default={})

    datapath = luigi.Parameter(default=".")
    column_function = luigi.Parameter(default=None)
    segments_filepath = luigi.Parameter(default=None)
    segmentation_threshold = luigi.FloatParameter(default=0.0005)
    tile_reduction_op = luigi.Parameter(default="mean")

    filters = luigi.OptionalParameter(default=None)

    filetype = luigi.Parameter(default="png")

    def requires(self):
        return AggregatedDatasetScenesAuxFieldWithEmbeddings(
            data_path=self.data_path,
            aux_name=self.aux_name,
            tiles_kind=self.tiles_kind,
            embedding_model_path=self.embedding_model_path,
            embedding_model_args=self.embedding_model_args,
            embedding_transform=self.embedding_transform,
            embedding_transform_args=self.embedding_transform_args,
            tile_reduction_op=self.tile_reduction_op,
        )
        # kwargs = dict(
        #     scalar_name=self.aux_name,
        #     emb_filepath=self.emb_filepath,
        #     datapath=self.datapath,
        #     column_function=self.column_function,
        #     segments_filepath=self.segments_filepath,
        #     segmentation_threshold=self.segmentation_threshold,
        #     reduce_op=self.reduce_op,
        #     filters=self.filters,
        # )

        # if isinstance(self.scene_ids, list) or isinstance(self.scene_ids, tuple):
        #     task = RegridEmbeddingsOnAuxField(scene_ids=list(self.scene_ids), **kwargs)
        # elif self.scene_ids == "available_data":
        #     base_task = RegridEmbeddingsOnAuxField(scene_ids=[], **kwargs)
        #     scene_ids = []
        #     for (scene_id, depd_task) in base_task.requires().items():
        #         task_src = depd_task.requires()["data"]
        #         if task_src.output().exists():
        #             scene_ids.append(scene_id)
        #     if len(scene_ids) == 0:
        #         raise Exception(f"No data available for `{self.aux_name}`")
        #     task = RegridEmbeddingsOnAuxField(scene_ids=scene_ids, **kwargs)
        # else:
        #     raise NotImplementedError(self.scene_ids)

        # return task

    def run(self):
        ds = self.input().open()
        self._ds = ds  # for plot titles etc
        var_name = f"{self.aux_name}__{self.tile_reduction_op}"
        scalar_dims = set(ds[var_name].dims)
        ds_stacked = ds.stack(sample=scalar_dims)

        emb_var = "emb"
        emb_dim = set(ds.dims).difference(set(ds[var_name].dims)).pop()

        # remove `explained_variance` since merging into a pandas DataFrame
        # later otherwise doesn't work (since the different components will
        # have different values)
        if "explained_variance" in ds_stacked:
            ds_stacked = ds_stacked.drop("explained_variance")

        self._make_plot(
            ds_stacked=ds_stacked, emb_var=emb_var, emb_dim=emb_dim, var_name=var_name
        )

    def _make_plot(self, ds_stacked, emb_var, emb_dim, var_name):
        raise NotImplementedError

    def _make_title(self):
        ds = self._ds
        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_path=self.embedding_model_path,
            **self.embedding_model_args,
        )

        if self.embedding_transform is not None:
            emb_name += "." + make_transform_name(
                transform=self.embedding_transform,
                **self.embedding_transform_args,
            )

        dataset_name = Path(self.datapath).absolute().stem
        n_scenes = len(np.unique(ds.scene_id.values))
        n_tiles = len(np.unique(ds.tile_id.values))
        title_parts = [
            f"{n_tiles} tiles across {n_scenes} scenes",
            f"{emb_name} from {dataset_name}",
        ]
        agg_parts = []
        if self.column_function is not None:
            agg_parts.append(self.column_function)

        agg_parts.append(self.tile_reduction_op)

        if self.segments_filepath is not None:
            segmentation_id = Path(self.segments_filepath).name.replace(".nc", "")
            agg_parts.append(f"{segmentation_id} segs")

        if getattr(self, "n_scalar_bins", None):
            agg_parts.append(f"{self.n_scalar_bins} bins")

        agg = " ".join(agg_parts)

        if len(agg) == 0:
            title_parts.insert(0, f"{self.aux_name}")
        else:
            title_parts.insert(0, f"{agg} of {self.aux_name}")

        if self.filters:
            title_parts.append(self.filters)

        title = "\n".join(title_parts)
        return title  # "\n".join(textwrap.wrap(text=title, width=40))

    def _build_output_name_parts(self):
        emb_name = make_embedding_name(
            kind=self.tiles_kind,
            model_path=self.embedding_model_path,
            **self.embedding_model_args,
        )
        name_parts = [self.aux_name, self.tile_reduction_op, "by", emb_name]

        if self.embedding_transform is not None:
            transform_model_args = dict(self.embedding_transform_args)
            if "pretrained_model" in transform_model_args:
                transform_name = transform_model_args["pretrained_model"].replace(
                    "/", "__"
                )
            else:
                transform_name = make_transform_name(
                    transform=self.embedding_transform,
                    **self.embedding_transform_args,
                )
            if transform_name != emb_name:
                name_parts.append("transformed")
                name_parts.append(transform_name)

        name_parts.append(self.plot_type)

        if self.filters is not None:
            name_parts.append(self.filters.replace("=", ""))

        if self.column_function is not None:
            name_parts.append(self.column_function)

        if self.segments_filepath is not None:
            name_parts.append(f"{self.segmentation_threshold}seg")

        return name_parts

    def output(self):
        name_parts = self._build_output_name_parts()
        identifier = ".".join(name_parts)
        fn = f"{identifier}.{self.filetype}"
        return luigi.LocalTarget(fn)


class ColumnScalarEmbeddingScatterPlot(AggPlotBaseTask):
    n_scalar_bins = luigi.IntParameter(default=None)
    cmap = luigi.Parameter(default="brg")
    plot_type = "scatter"

    def reduce(self, da):
        try:
            return getattr(da, self.tile_reduction_op)()
        except AttributeError:
            raise NotImplementedError(self.tile_reduction_op)

    def _build_output_name_parts(self):
        name_parts = super()._build_output_name_parts()

        if self.n_scalar_bins is not None:
            name_parts.append(f"{self.n_scalar_bins}bins")

        return name_parts

    def _make_plot(self, ds_stacked, emb_var, emb_dim):
        if self.n_scalar_bins is not None:
            bins = np.linspace(
                ds_stacked[self.aux_name].min(),
                ds_stacked[self.aux_name].max(),
                self.n_scalar_bins,
            )
            bin_labels = 0.5 * (bins[1:] + bins[:-1])
            ds_stacked = self.reduce(
                ds_stacked.groupby_bins(self.aux_name, bins=bins, labels=bin_labels)
            )

        data = {
            f"{emb_dim}=0": ds_stacked[emb_var].sel({emb_dim: 0}),
            f"{emb_dim}=1": ds_stacked[emb_var].sel({emb_dim: 1}),
            self.aux_name: ds_stacked[self.aux_name],
        }

        df = pd.DataFrame(data, columns=data.keys())

        fig, ax = plt.subplots(figsize=(4, 5))
        sns.scatterplot(
            x=f"{emb_dim}=0",
            y=f"{emb_dim}=1",
            ax=ax,
            hue=self.aux_name,
            data=df,
            palette=self.cmap,
        )
        if self.segments_filepath is not None or self.n_scalar_bins is not None:
            ax.set_xlim(-3.0e-1, 4.0e-1)
            ax.set_ylim(-3.0e-1, 4.0e-1)

        ax.set_aspect(1.0)
        title = self._make_title()
        ax.set_title(title + "\n")
        sns.despine(ax=ax)
        fig.tight_layout()
        # fig.subplots_adjust(top=0.8)
        fig.savefig(self.output().fn)


class ColumnScalarEmbeddingContourPlot(AggPlotBaseTask):
    n_scalar_bins = luigi.IntParameter()
    contour_level = luigi.Parameter(default=0.1)
    plot_type = "contour"

    def _make_plot(self, ds_stacked, emb_var, emb_dim):

        fig, ax = plt.subplots(figsize=(10, 10))
        ax_inset = ax.inset_axes([0.1, 0.8, 0.2, 0.2])

        def _create_kde(ds_subset, var_name, level=0.1, **kwargs):
            da = ds_subset[var_name]
            label = f"{da.min().item():.4g} - {da.max().item():.4g}"

            data = {
                f"{emb_dim}=0": ds_subset[emb_var].sel({emb_dim: 0}),
                f"{emb_dim}=1": ds_subset[emb_var].sel({emb_dim: 1}),
                self.aux_name: ds_subset[var_name],
            }
            df = pd.DataFrame(data, columns=data.keys())

            return sns.kdeplot(
                x=f"{emb_dim}=0",
                y=f"{emb_dim}=1",
                data=df,
                ax=ax,
                levels=[level],
                label=label,
                **kwargs,
            )

        def ecdf(x):
            x_ = np.sort(x)
            x_ = x_[~np.logical_or(np.isnan(x_), np.isinf(x_))]
            N_points = len(x_)
            y_ = (np.arange(N_points) + 1.0) / N_points

            return x_, y_

        def ecdf_da(da):
            x_, y_ = ecdf(da)
            return xr.DataArray(y_, coords={da.name: x_}, dims=(da.name))

        def all_cumulative_bins(da, n_bins=3):
            da_ecdf = ecdf_da(da=da)

            for v in np.linspace(0.0, 1.0, n_bins + 1):
                i = (np.abs(da_ecdf - v)).argmin()
                da_pt = da_ecdf.isel({da.name: i})
                yield da_pt[da.name], da_pt

        def find_cumulative_bins(da, n_bins=3):
            x_bins, bin_fractions = zip(*all_cumulative_bins(da=da, n_bins=n_bins))
            return x_bins, xr.concat(bin_fractions, dim=da.name)

        n_bins = self.n_scalar_bins
        var_name = self.aux_name
        ds_ = ds_stacked

        bins, da_bin_fractions = find_cumulative_bins(da=ds_[var_name], n_bins=n_bins)
        da_ecdf = ecdf_da(da=ds_[var_name])

        data_groups = list(ds_.groupby_bins(var_name, bins=bins))
        data_groups = sorted(data_groups, key=lambda dg: dg[0].left)

        da_ecdf.plot(ax=ax_inset, color="black")
        for (dg_interval, ds_group) in data_groups:
            f_left = da_bin_fractions.sel(
                {var_name: dg_interval.left}, method="nearest"
            )
            f_right = da_bin_fractions.sel(
                {var_name: dg_interval.right}, method="nearest"
            )
            f_avg = 0.5 * (f_left + f_right)
            # line, = ax_inset.plot([dg_interval.left, dg_interval.right], [f_avg, f_avg])
            # use errorbar so we can get caps on the ends of the lines to show the range
            line, _, _ = ax_inset.errorbar(
                0.5 * (dg_interval.left + dg_interval.right),
                f_avg,
                xerr=(dg_interval.right - dg_interval.left) / 2.0,
                capsize=3,
            )
            _create_kde(
                ds_subset=ds_group,
                var_name=var_name,
                color=line.get_color(),
                level=self.contour_level,
            )

        ax_inset.set_ylabel("cdf [1]")
        ax_inset.set_xlabel(xr.plot.utils.label_from_attrs(ds_[var_name]))
        ax.legend(title=xr.plot.utils.label_from_attrs(ds_[var_name]))

        # ax.set_xlim(-3.0e-1, 4.0e-1)
        # ax.set_ylim(-3.0e-1, 4.0e-1)
        sns.despine(ax=ax)
        sns.despine(ax=ax_inset)

        title = self._make_title()
        ax.set_title(title + "\n")

        fig.savefig(self.output().fn)

    def _build_output_name_parts(self):
        name_parts = super()._build_output_name_parts()
        name_parts.append(f"{self.n_scalar_bins}bins{self.contour_level}")
        return name_parts


class ColumnScalarEmbeddingDistPlot(AggPlotBaseTask):
    """
    Plot the distribution of a scalar as a 2D histogram using the first two
    dimensions of the provided embedding vectors
    """

    cmap = luigi.Parameter(default="brg")
    plot_type = "hist2d"
    statistics = luigi.OptionalListParameter(
        default=["mean", "sem", "min", "max", "std", "median"]
    )
    dx = luigi.OptionalFloatParameter(default=0.1)
    dy = luigi.OptionalParameter(default="dx")

    def _make_plot(self, ds_stacked, emb_var, emb_dim, var_name):
        data = {
            f"{emb_dim}=0": ds_stacked[emb_var].sel({emb_dim: 0}),
            f"{emb_dim}=1": ds_stacked[emb_var].sel({emb_dim: 1}),
            var_name: ds_stacked[var_name],
        }
        ds = xr.Dataset(data)
        ds[f"{emb_dim}=0"].attrs["units"] = "1"
        ds[f"{emb_dim}=1"].attrs["units"] = "1"

        dx = self.dx
        if self.dy == "dx":
            dy = dx

        fig, _ = plot_types.dist_plots(
            ds=ds,
            x=f"{emb_dim}=0",
            y=f"{emb_dim}=1",
            v=var_name,
            dx=dx,
            dy=dy,
            cmap="jet",
            statistics=self.statistics,
        )
        title = self._make_title()
        fig.suptitle(title + "\n", y=1.04)
        fig.savefig(self.output().fn, bbox_inches="tight")


class ColumnScalarEmbeddingScatterPlotCollection(luigi.WrapperTask):
    emb_filepath = luigi.Parameter()
    segments_filepath = luigi.Parameter(default=None)
    plot_type = luigi.Parameter()
    datapath = luigi.Parameter()

    def _build_tasks(self, task_kwargs, TaskClass):
        levels = [110, 130]
        u_vars = ["u", "v", "umag"]

        tasks = []
        for u_var in u_vars:
            for level in levels:
                t = TaskClass(
                    column_function=f"level__{level}", scalar_name=u_var, **task_kwargs
                )
                tasks.append(t)

        for level in levels:
            t = TaskClass(
                column_function=f"level__{level}", scalar_name="q", **task_kwargs
            )
            tasks.append(t)

        t = TaskClass(scalar_name="d_theta__lts", **task_kwargs)
        tasks.append(t)
        t = TaskClass(scalar_name="d_theta__eis", **task_kwargs)
        tasks.append(t)
        t = TaskClass(scalar_name="z_lcl", **task_kwargs)
        tasks.append(t)
        t = TaskClass(scalar_name="sst", **task_kwargs)
        tasks.append(t)

        t = TaskClass(
            column_function="leveldiff__130__110", scalar_name="q", **task_kwargs
        )
        tasks.append(t)
        return tasks

    def requires(self):
        filetype = "png"

        task_kwargs = dict(
            emb_filepath=self.emb_filepath,
            segments_filepath=self.segments_filepath,
            filetype=filetype,
            datapath=self.datapath,
        )
        tasks = []

        if self.plot_type == "scatter":
            TaskClass = ColumnScalarEmbeddingScatterPlot
            for n_scalar_bins in [None, 40]:
                task_kwargs["n_scalar_bins"] = n_scalar_bins
                tasks += self._build_tasks(task_kwargs=task_kwargs, TaskClass=TaskClass)
            task_kwargs = dict()
        elif self.plot_type == "dist":
            TaskClass = ColumnScalarEmbeddingContourPlot
            task_kwargs["n_scalar_bins"] = 10
            tasks += self._build_tasks(task_kwargs=task_kwargs, TaskClass=TaskClass)
        elif self.plot_type == "hist2d":
            TaskClass = ColumnScalarEmbeddingDistPlot
            tasks += self._build_tasks(task_kwargs=task_kwargs, TaskClass=TaskClass)
        else:
            raise NotImplementedError(self.plot_type)

        return tasks


class ColumnScalarEmbeddingScatterPlotSuite(luigi.WrapperTask):
    emb_filepath = luigi.Parameter()
    segments_filepath = luigi.Parameter(default=None)
    datapath = luigi.Parameter()
    plot_types = luigi.ListParameter()

    def requires(self):
        tasks = []
        for plot_type in self.plot_types:
            tasks.append(
                ColumnScalarEmbeddingScatterPlotCollection(
                    emb_filepath=self.emb_filepath,
                    segments_filepath=self.segments_filepath,
                    plot_type=plot_type,
                    datapath=self.datapath,
                )
            )

        return tasks
