import luigi
import matplotlib.pyplot as plt

from .. import DataSource
from ..sources.goes16.satpy_rgb import rgb_da_to_img
from .regridding import SceneRegriddedData


def align_axis_x(ax, ax_target):
    """Make x-axis of `ax` aligned with `ax_target` in figure"""
    posn_old, posn_target = ax.get_position(), ax_target.get_position()
    ax.set_position([posn_target.x0, posn_old.y0, posn_target.width, posn_old.height])


def get_aux_dataarray(datasource, scene_id, aux_name):
    task = SceneRegriddedData(
        data_path=datasource.data_path, scene_id=scene_id, aux_name=aux_name
    )
    da_aux = task.output()["data"].open()
    return da_aux


def get_scene_img(datasource, scene_id):
    da_scene = get_aux_dataarray(
        datasource=datasource, aux_name=None, scene_id=scene_id
    )
    da_scene.attrs.update(dict(standard_name="true_color"))
    img = rgb_da_to_img(da_scene).pil_image()
    return img


def create_rect_aux_plot(datasource, scene_id, aux_variables, fig_width=12.0):
    aux_dataarrays = {
        v: get_aux_dataarray(datasource=datasource, aux_name=v, scene_id=scene_id)
        for v in aux_variables
    }

    img = get_scene_img(datasource=datasource, scene_id=scene_id)
    aspect = img.size[0] / img.size[1]

    crs = datasource.domain.crs
    n_aux = len(aux_variables)
    n_rows = n_aux + 1
    col_height = fig_width / aspect
    fig, axes = plt.subplots(
        nrows=n_rows,
        figsize=(fig_width, col_height * n_rows),
        subplot_kw=dict(projection=crs),
        sharex=True,
    )

    ax = axes[0]
    ax.imshow(
        img, transform=datasource.domain.crs, extent=datasource.domain.get_grid_extent()
    )
    ax.set_title(f"truecolor rgb - {scene_id}")

    for ax, aux_name in zip(axes[1:], aux_variables):
        da_aux = aux_dataarrays[aux_name]
        da_aux.plot(ax=ax, rasterized=True, transform=crs)
        ax.set_title(f"{aux_name} - {scene_id}")

    for ax in axes:
        ax.coastlines(color="white")
        ax.gridlines(draw_labels=["left", "bottom"])

    fig.tight_layout()
    align_axis_x(ax=axes[0], ax_target=axes[1])

    return fig, axes


class AuxRectScenePlot(luigi.Task):
    data_path = luigi.Parameter()
    scene_id = luigi.Parameter()
    aux_variables = luigi.ListParameter()
    fig_width = luigi.FloatParameter(default=12.0)

    def requires(self):
        tasks = {}
        tasks["truecolor_rgb"] = SceneRegriddedData(
            data_path=self.data_path, scene_id=self.scene_id
        )

        for v in self.aux_variables:
            tasks[v] = SceneRegriddedData(
                data_path=self.data_path, scene_id=self.scene_id, aux_name=v
            )

        return tasks

    def run(self):
        datasource = DataSource.load(self.data_pat)
        fig, axes = create_rect_aux_plot(
            datasource=datasource,
            scene_id=self.scene_id,
            aux_variables=self.aux_variables,
            fig_width=self.fig_width,
        )
        fig.savefig(self.output().path)

    def output(self):
        fn = f"{self.scene_id}.pdf"
        return luigi.LocalTarget(fn)
