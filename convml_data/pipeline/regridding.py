from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import regridcart as rc

from .. import DataSource
from ..sampling import domain as sampling_domain
from ..utils.domain_images import align_axis_x
from ..utils.luigi import ImageTarget, XArrayTarget
from .aux_sources import AuxTaskMixin, CheckForAuxiliaryFiles
from .sampling import CropSceneSourceFiles, SceneSourceFiles, _SceneRectSampleBase
from .scene_images import SceneImageMixin
from .scene_sources import GenerateSceneIDs
from .utils import SceneBulkProcessingBaseTask


def _plot_scene_aux(da_aux, img, **kwargs):
    fig, axes = plt.subplots(nrows=2, figsize=(10.0, 8.0), sharex=True)
    img_extent = [
        da_aux.x.min().item(),
        da_aux.x.max().item(),
        da_aux.y.min().item(),
        da_aux.y.max().item(),
    ]
    axes[0].imshow(img, extent=img_extent)
    axes[1].imshow(img, extent=img_extent)
    if da_aux.name == "HT":
        # cloud-top height [m]
        kwargs["vmin"] = 0.0
        kwargs["vmax"] = 14.0e3
    da_aux.plot(ax=axes[1], y="y", cmap="nipy_spectral", **kwargs)
    align_axis_x(ax=axes[0], ax_target=axes[1])
    return fig, axes


class SceneRegriddedData(_SceneRectSampleBase, SceneImageMixin, AuxTaskMixin):
    """
    Regrid the scene source data to a fixed Cartesian resolution
    """

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        data_source = self.data_source

        reqs = {}
        if isinstance(data_source.domain, sampling_domain.SourceDataDomain):
            reqs["source_data"] = SceneSourceFiles(
                scene_id=self.scene_id,
                data_path=self.data_path,
                aux_name=self.aux_name,
            )
        else:
            reqs["source_data"] = CropSceneSourceFiles(
                scene_id=self.scene_id,
                data_path=self.data_path,
                pad_ptc=self.crop_pad_ptc,
                aux_name=self.aux_name,
            )

        if self.aux_name is not None:
            reqs["base"] = SceneRegriddedData(
                scene_id=self.scene_id,
                data_path=self.data_path,
                crop_pad_ptc=self.crop_pad_ptc,
            )

        return reqs

    @property
    def scene_resolution(self):
        data_source = self.data_source
        if (
            "resolution" not in data_source.sampling
            or data_source.sampling.get("resolution") is None
        ):
            raise Exception(
                "To produce isometric grid resampling of the source data please "
                "define the grid-spacing by defining `resolution` (in meters/pixel) "
                "in the `sampling` part of the data source meta information"
            )
        return data_source.sampling["resolution"]

    def run(self):
        domain_output = self.output()

        inputs = self.input()
        da_src = inputs["source_data"]["data"].open()

        if not domain_output["data"].exists():
            domain = self.data_source.domain
            if isinstance(domain, sampling_domain.SourceDataDomain):
                domain = domain.generate_from_dataset(ds=da_src)

            dx = self.scene_resolution

            if self.aux_name is None:
                method = "bilinear"
            else:
                method = "nearest_s2d"

            da_domain = rc.resample(
                domain=domain,
                da=da_src,
                dx=dx,
                method=method,
                keep_attrs=True,
                apply_crop=False,
            )

            assert "units" in da_domain.attrs
            assert "long_name" in da_domain.attrs

            Path(domain_output["data"].fn).parent.mkdir(exist_ok=True, parents=True)
            domain_output["data"].write(da_domain)
        else:
            da_domain = domain_output["data"].open()

        if "image" in self.output():
            img_domain = self._create_image(da_scene=da_domain)
            img_domain.save(str(domain_output["image"].fn))

    def output(self):
        scene_data_path = Path(self.data_path) / "rect"

        if self.aux_name is not None:
            scene_data_path = scene_data_path / "aux" / self.aux_name

        fn_data = f"{self.scene_id}.nc"
        outputs = dict(
            data=XArrayTarget(str(scene_data_path / fn_data)),
        )

        if self.image_function is not None:
            fn_image = f"{self.scene_id}.png"
            outputs["image"] = ImageTarget(str(scene_data_path / fn_image))
        return outputs


class GenerateRegriddedScenes(SceneBulkProcessingBaseTask):
    data_path = luigi.Parameter(default=".")
    TaskClass = SceneRegriddedData

    aux_name = luigi.OptionalParameter(default=None)

    def _get_task_class_kwargs(self, scene_ids):
        if self.aux_name is None:
            return {}

        return dict(aux_name=self.aux_name)

    def requires(self):
        if self.TaskClass is None:
            raise Exception(
                "Please set TaskClass to the type you would like"
                " to process for every scene"
            )
        if self.aux_name is None:
            TaskClass = GenerateSceneIDs
        else:
            TaskClass = CheckForAuxiliaryFiles

        return TaskClass(data_path=self.data_path, **self._get_scene_ids_task_kwargs())

    def _get_scene_ids_task_kwargs(self):
        if self.aux_name is None:
            return {}

        return dict(aux_name=self.aux_name)
