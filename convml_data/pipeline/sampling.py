"""
pipeline tasks related to spatial cropping, reprojection and sampling of scene data
"""
from pathlib import Path

import luigi

from .. import DataSource
from ..sources import build_fetch_tasks, extract_variable
from ..utils.luigi import ImageTarget, XArrayTarget
from .aux_sources import CheckForAuxiliaryFiles
from .scene_images import SceneImageMixin
from .scene_sources import GenerateSceneIDs
from .utils import SceneBulkProcessingBaseTask


class SceneSourceFiles(luigi.Task):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    aux_name = luigi.OptionalParameter(default=None)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        if self.aux_name is not None:
            return CheckForAuxiliaryFiles(
                data_path=self.data_path, aux_name=self.aux_name
            )
        else:
            return GenerateSceneIDs(data_path=self.data_path)

    def _build_fetch_tasks(self):
        task = None
        ds = self.data_source

        if self.aux_name is None:
            source_name = ds.source
        else:
            source_name = self.data_source.aux_products[self.aux_name]["source"]
        source_data_path = Path(self.data_path) / "source_data" / source_name

        if self.input().exists():
            all_source_files = self.input().open()
            if self.scene_id not in all_source_files:
                raise Exception(
                    f"Data for scene {self.scene_id} not found"
                    + (self.aux_name is not None and f" for {self.aux_name}" or "")
                )
            scene_source_files = all_source_files[self.scene_id]
            task = build_fetch_tasks(
                scene_source_files=scene_source_files,
                source_name=source_name,
                source_data_path=source_data_path,
            )

        return task

    def run(self):
        fetch_tasks = self._build_fetch_tasks()
        if fetch_tasks is not None:
            yield fetch_tasks

    def output(self):
        source_tasks = self._build_fetch_tasks()
        if source_tasks is not None:
            return {
                input_name: source_task.output()
                for (input_name, source_task) in source_tasks.items()
            }
        else:
            return luigi.LocalTarget(f"__fake_target__{self.scene_id}__")


class CropSceneSourceFiles(luigi.Task, SceneImageMixin):
    scene_id = luigi.Parameter()
    data_path = luigi.Parameter(default=".")
    pad_ptc = luigi.FloatParameter(default=0.1)
    aux_name = luigi.OptionalParameter(default=None)

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        return dict(
            data=SceneSourceFiles(
                data_path=self.data_path,
                scene_id=self.scene_id,
                aux_name=self.aux_name,
            )
        )

    @property
    def domain(self):
        return self.data_source.domain

    def run(self):
        data_source = self.data_source

        if self.aux_name is None:
            source_name = data_source.source
            product_name = data_source.product
            product_identifier = self.data_source.name
            product_meta = dict(name=product_identifier)
            if product_name == "user_function":
                if "input" not in self.data_source._meta:
                    raise Exception(
                        "To use a user-function on the primary data-set being"
                        " produced you need to define the channels used by defining"
                        " the section `input` in the input `meta.yaml` file"
                    )
                else:
                    product_meta["input"] = self.data_source._meta["input"]
        else:
            source_name = self.data_source.aux_products[self.aux_name]["source"]
            product_name = self.data_source.aux_products[self.aux_name]["product"]
            product_identifier = self.aux_name
            product_meta = self.data_source.aux_products[self.aux_name]

        product_meta["context"] = dict(
            datasource_path=self.data_path,
            product_identifier=product_identifier,
        )

        domain = self.domain

        da_cropped = extract_variable(
            task_input=self.input()["data"],
            data_source=source_name,
            product=product_name,
            domain=domain,
            product_meta=product_meta,
        )

        if "image" in self.output():
            img_cropped = self._create_image(da_scene=da_cropped)

        self.output_path.mkdir(exist_ok=True, parents=True)
        self.output()["data"].write(da_cropped)

        if "image" in self.output():
            self.output()["image"].write(img_cropped)

    @property
    def output_path(self):
        ds = self.data_source

        output_path = Path(self.data_path) / "source_data"

        if self.aux_name is None:
            output_path = output_path / ds.source / ds.product
        else:
            product_name = self.data_source.aux_products[self.aux_name]["product"]
            source_name = self.data_source.aux_products[self.aux_name]["source"]
            if product_name == "user_function":
                product_name = f"{product_name}__{self.aux_name}"
            output_path = output_path / source_name / product_name

        output_path = output_path / "cropped"

        return output_path

    def output(self):
        data_path = self.output_path

        fn_data = f"{self.scene_id}.nc"
        outputs = dict(data=XArrayTarget(str(data_path / fn_data)))

        if self.image_function is not None:
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
    aux_name = luigi.OptionalParameter(default=None)

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
            aux_name=self.aux_name,
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

    aux_name = luigi.OptionalParameter(default=None)

    def _get_scene_ids_task_class(self):
        if self.aux_name is None:
            return GenerateSceneIDs
        else:
            return CheckForAuxiliaryFiles

    def _get_task_class_kwargs(self, scene_ids):
        return dict(aux_name=self.aux_name)

    def _get_scene_ids_task_kwargs(self):
        if self.aux_name is None:
            return {}
        return dict(aux_name=self.aux_name)
