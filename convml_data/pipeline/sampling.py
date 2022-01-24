"""
pipeline tasks related to spatial cropping, reprojection and sampling of scene data
"""
from pathlib import Path

import luigi
import regridcart as rc

from .. import DataSource
from ..sources import build_fetch_tasks, create_image, extract_variable
from ..utils.luigi import ImageTarget, XArrayTarget
from .aux_sources import CheckForAuxiliaryFiles
from .scene_sources import GenerateSceneIDs
from .utils import SceneBulkProcessingBaseTask


class SceneSourceFiles(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    aux_product = luigi.OptionalParameter(default=None)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        if self.aux_product is not None:
            return CheckForAuxiliaryFiles(
                data_path=self.data_path, product_name=self.aux_product
            )
        else:
            return GenerateSceneIDs(data_path=self.data_path)

    def _build_fetch_tasks(self):
        task = None
        ds = self.data_source
        source_data_path = Path(self.data_path) / "source_data" / ds.source

        if self.input().exists():
            all_source_files = self.input().open()
            scene_source_files = all_source_files[self.scene_id]
            task = build_fetch_tasks(
                scene_source_files=scene_source_files,
                source_name=ds.source,
                source_data_path=source_data_path,
            )

        return task

    def run(self):
        fetch_tasks = self._build_fetch_tasks()
        if fetch_tasks is not None:
            yield fetch_tasks

    def output(self):
        source_task = self._build_fetch_tasks()
        if source_task is not None:
            return source_task.output()
        else:
            return luigi.LocalTarget(f"__fake_target__{self.scene_id}__")


class CropSceneSourceFiles(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    pad_ptc = luigi.FloatParameter(default=0.1)
    aux_product = luigi.OptionalParameter(default=None)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return dict(
            data=SceneSourceFiles(
                data_path=self.data_path,
                scene_id=self.scene_id,
                aux_product=self.aux_product,
            )
        )

    @property
    def domain(self):
        data_source = self.data_source
        domain = data_source.domain
        ds_input = self.input().open()
        if isinstance(domain, rc.LocalCartesianDomain):
            domain.validate_dataset(ds=ds_input)
        return domain

    def run(self):
        data_source = self.data_source

        if self.aux_product is not None:
            product = self.aux_product
        else:
            product = data_source.type

        da_full = extract_variable(
            task_input=self.input(),
            data_source=data_source.source,
            product=product,
        )
        domain = data_source.domain
        if isinstance(domain, rc.LocalCartesianDomain):
            domain.validate_dataset(da_full)

        da_cropped = rc.crop_field_to_domain(
            domain=self.domain, da=da_full, pad_pct=self.pad_ptc
        )

        img_cropped = None
        if data_source.source == "goes16" and product == "truecolor_rgb":
            # to be able to create a RGB image with satpy we need to set the
            # attrs again to ensure we get a proper RGB image
            da_cropped.attrs.update(da_full.attrs)
            img_cropped = create_image(
                da=da_cropped, data_source=data_source.source, product=product
            )

        self.output_path.mkdir(exist_ok=True, parents=True)
        self.output()["data"].write(da_cropped)

        if img_cropped is not None:
            self.output()["image"].write(img_cropped)

    @property
    def output_path(self):
        ds = self.data_source

        output_path = Path(self.data_path) / "source_data" / ds.source

        if self.aux_product is None:
            output_path = output_path / ds.type
        else:
            output_path = output_path / "aux" / self.aux_product

        output_path = output_path / "cropped"

        return output_path

    def output(self):
        data_path = self.output_path

        fn_data = f"{self.scene_id}.nc"
        outputs = dict(data=XArrayTarget(str(data_path / fn_data)))

        data_source = self.data_source
        if data_source.source == "goes16" and self.aux_product is None:
            if data_source.type == "truecolor_rgb":
                fn_image = f"{self.scene_id}.png"
                outputs["image"] = ImageTarget(str(data_path / fn_image))

        return outputs


class _SceneRectSampleBase(luigi.Task):
    """
    This task represents the creation of scene data to feed into a neural
    network, either the data for an entire rectangular domain or a single tile
    """

    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    crop_pad_ptc = luigi.FloatParameter(default=0.1)
    aux_product = luigi.OptionalParameter(default=None)

    def requires(self):
        t_scene_ids = GenerateSceneIDs(data_path=self.data_path)
        if not t_scene_ids.output().exists():
            raise Exception(
                "Scene IDs haven't been defined for this dataset yet "
                "Please run the `GenerateSceneIDs task first`"
            )
        # all_scenes = t_scene_ids.output().open()
        # source_channels = all_scenes[self.scene_id]

        return CropSceneSourceFiles(
            scene_id=self.scene_id,
            data_path=self.data_path,
            pad_ptc=self.crop_pad_ptc,
            aux_product=self.aux_product,
        )


class SceneRectData(_SceneRectSampleBase):
    def output(self):
        scene_data_path = Path(self.data_path) / "scenes"
        fn_data = f"{self.scene_id}.nc"
        fn_image = f"{self.scene_id}.png"
        return dict(
            data=XArrayTarget(str(scene_data_path / fn_data)),
            image=XArrayTarget(str(scene_data_path / fn_image)),
        )


class GenerateCroppedScenes(SceneBulkProcessingBaseTask):
    data_path = luigi.Parameter(default=".")
    TaskClass = CropSceneSourceFiles

    aux_product = luigi.OptionalParameter(default=None)

    def _get_scene_ids_task_class(self):
        if self.aux_product is None:
            return GenerateSceneIDs
        else:
            return CheckForAuxiliaryFiles

    def _get_task_class_kwargs(self, scene_ids):
        return dict(aux_product=self.aux_product)

    def _get_scene_ids_task_kwargs(self):
        if self.aux_product is None:
            return {}
        return dict(product_name=self.aux_product)
