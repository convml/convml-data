from pathlib import Path

import luigi
import numpy as np

from .. import DataSource
from ..sources import build_query_tasks
from ..utils.luigi import DBTarget
from .scene_sources import GenerateSceneIDs, get_time_for_filename, parse_scene_id


class CheckForAuxiliaryFiles(luigi.Task):
    """
    Convenience task for downloading extra data for each scene.
    """

    data_path = luigi.Parameter(default=".")
    aux_name = luigi.Parameter()
    DT_MAX_DEFAULT = np.timedelta64(10, "m")

    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    def requires(self):
        tasks = {}
        tasks["scene_ids"] = GenerateSceneIDs(data_path=self.data_path)

        aux_source_name = self.aux_source_name
        aux_product_name = self.aux_product_name
        source_data_path = Path(self.data_path) / "source_data" / aux_source_name

        tasks["product"] = build_query_tasks(
            source_name=aux_source_name,
            source_type=aux_product_name,
            source_data_path=source_data_path,
            time_intervals=self.data_source.time_intervals,
        )

        return tasks

    @property
    def aux_product_meta(self):
        product_meta = self.data_source.aux_products.get(self.aux_name)
        if product_meta is None:
            raise Exception(
                f"Please define `{self.aux_name}` in the `aux_products`"
                " group in meta.yaml"
            )

        return product_meta

    @property
    def dt_max(self):
        return self.aux_product_meta.get("dt_max", self.DT_MAX_DEFAULT)

    @property
    def aux_source_name(self):
        return self.aux_product_meta["source"]

    @property
    def aux_product_name(self):
        return self.aux_product_meta["type"]

    def run(self):
        """
        Map the times for which aux data is available to the scene IDs we
        already have
        """
        inputs = self.input()

        scene_ids = list(inputs.pop("scene_ids").open().keys())
        scene_times = np.array([parse_scene_id(scene_id)[1] for scene_id in scene_ids])

        aux_source_name = self.aux_source_name

        product_fn_for_scenes = {}
        dt_min = np.timedelta64(365, "D")
        for product_inputs in inputs["product"]:
            aux_product_filenames = product_inputs.open()
            aux_times = np.array(
                [
                    get_time_for_filename(source_name=aux_source_name, filename=fn)
                    for fn in aux_product_filenames
                ]
            )
            for aux_filename, aux_time in zip(aux_product_filenames, aux_times):
                dt_all = np.abs(scene_times - aux_time)
                i = np.argmin(dt_all)
                dt = dt_all[i]
                if dt <= self.dt_max:
                    scene_id = scene_ids[i]
                    product_fn_for_scenes[scene_id] = aux_filename
                if dt < dt_min:
                    dt_min = dt

        if len(product_fn_for_scenes) == 0:
            raise Exception(
                f"No aux scenes found within the time-window `{self.dt_max}`"
                f" dt_min=`{dt_min}`. Try changing `dt_max`"
            )

        self.output().write(product_fn_for_scenes)

    def output(self):
        data_source = self.data_source
        output = None
        aux_source_name = self.aux_source_name
        aux_product_name = self.aux_product_name

        if data_source.source == "goes16":
            p = (
                Path(self.data_path)
                / "source_data"
                / aux_source_name
                / aux_product_name
            )
            output = DBTarget(
                path=str(p), db_name=aux_product_name, db_type=data_source.db_type
            )

        return output
