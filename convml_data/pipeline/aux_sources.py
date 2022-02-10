from pathlib import Path

import isodate
import luigi
import numpy as np

from .. import DataSource
from ..sources import build_query_tasks
from ..utils.luigi import DBTarget
from .scene_sources import (
    GenerateSceneIDs,
    create_scenes_from_multichannel_queries,
    get_time_for_filename,
    parse_scene_id,
)


class CheckForAuxiliaryFiles(luigi.Task):
    """
    Convenience task for downloading extra data for each scene and matching the
    available data to the scenes of the primary datasource
    """

    data_path = luigi.Parameter(default=".")
    aux_name = luigi.Parameter()

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

        if "scene_mapping_strategy" not in product_meta:
            product_meta["scene_mapping_strategy"] = "single_scene_per_aux_time"

        return product_meta

    @property
    def dt_aux(self):
        dt_aux = self.aux_product_meta.get("dt_aux", None)
        if dt_aux is not None:
            dt_aux = isodate.parse_duration(dt_aux)
        return dt_aux

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

        product_input = inputs["product"]

        # create a mapping from aux_scene_time -> aux_scene_filename(s)
        aux_source_name = self.aux_source_name
        aux_product_name = self.aux_product_name
        if type(product_input) == dict:
            aux_scenes_by_time = create_scenes_from_multichannel_queries(
                inputs=product_input,
                source_name=aux_source_name,
                product=aux_product_name,
            )
        else:
            aux_scenes_by_time = {}
            for product_inputs in inputs["product"]:
                aux_product_filenames = product_inputs.open()
                aux_times = np.array(
                    [
                        get_time_for_filename(source_name=aux_source_name, filename=fn)
                        for fn in aux_product_filenames
                    ]
                )
                for (t, filename) in zip(aux_times, aux_product_filenames):
                    aux_scenes_by_time[t] = filename

        product_fn_for_scenes = _match_each_aux_time_to_scene_ids(
            aux_scenes_by_time=aux_scenes_by_time,
            scene_times=scene_times,
            scene_ids=scene_ids,
            dt_aux=self.dt_aux,
            strategy=self.aux_product_meta["scene_mapping_strategy"],
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


def _match_each_aux_time_to_scene_ids(
    aux_scenes_by_time,
    scene_times,
    scene_ids,
    strategy="single_scene_per_aux_time",
    dt_aux=None,
):
    """
    Match the aux source-file(s) to each of the primary scene IDs we've got,
    using the time-spacing in aux source-file(s) `dt_aux` (if `dt_aux == None`
    it will be calculated as the smallest time separation between aux
    source-file(s))

    Two strategies are implemented:

    1) `single_scene_per_aux_time`: Each aux source will be matched to a single
       scene id by selecting the closest scene id (in time). A limit on the
       separation in time is set as half the time-spacing between the aux
       sources `dt_aux/2`

    2) `all_scenes_within_dt_aux`: All scene ids within `dt_aux/2` of the
       time of one aux source-file(s) time are matched to the same aux
       source-file(s)
    """
    aux_times = np.array(list(aux_scenes_by_time.keys()))
    dt_aux_all = np.diff(aux_times)
    # we take the minimum time here because there could be gaps in the aux data
    # (if we've only downloaded data over time intervals)
    dt_aux = np.min(dt_aux_all)
    dt_max = dt_aux / 2

    product_fn_for_scenes = {}

    if strategy == "single_scene_per_aux_time":
        # now match these aux scene times with the scene IDs we've already got
        # by finding the closest scene each time
        dt_min = np.timedelta64(365, "D")
        for aux_time, aux_scene in aux_scenes_by_time.items():
            dt_all = np.abs(scene_times - aux_time)
            i = np.argmin(dt_all)
            dt = dt_all[i]
            if dt <= dt_max:
                scene_id = scene_ids[i]
                product_fn_for_scenes[scene_id] = aux_scene
            if dt < dt_min:
                dt_min = dt
    elif strategy == "all_scenes_within_dt_aux":
        for scene_time, scene_id in zip(scene_times, scene_ids):
            dt_all = np.abs(aux_times - scene_time)
            i = np.argmin(dt_all)
            dt = dt_all[i]
            if dt <= dt_max:
                aux_time = aux_times[i]
                product_fn_for_scenes[scene_id] = aux_scenes_by_time[aux_time]
    else:
        raise NotImplementedError(strategy)

    if len(product_fn_for_scenes) == 0:
        raise Exception(
            "No aux scenes found within the inferred time-resolution of the "
            f"aux data `{dt_aux}`"
        )

    return product_fn_for_scenes
