from pathlib import Path

import isodate
import luigi
import numpy as np

from .. import DataSource
from ..sources import IncompleteSourceDefition, build_query_tasks
from ..utils.luigi import DBTarget
from .scene_sources import (
    GenerateSceneIDs,
    create_scenes_from_input_queries,
    parse_scene_id,
)

# need to use three underscores because some product names might have two underscores in them
EXTRA_PRODUCT_SEPERATOR = "___"
EXTRA_PRODUCT_SENTINEL = f"{EXTRA_PRODUCT_SEPERATOR}extra{EXTRA_PRODUCT_SEPERATOR}"


class AuxTaskMixin(object):
    @property
    def data_source(self):
        return DataSource.load(path=self.data_path)

    @property
    def source_name(self):
        if self.aux_name is None:
            source_name = self.data_source.source
        elif self.aux_name.startswith(EXTRA_PRODUCT_SENTINEL):
            *_, source_name, _ = self.aux_name.split(EXTRA_PRODUCT_SEPERATOR)
        else:
            source_name = self.data_source.aux_products[self.aux_name]["source"]
        return source_name

    @property
    def product_name(self):
        if self.aux_name is None:
            product_name = self.data_source.product
        elif self.aux_name.startswith(EXTRA_PRODUCT_SEPERATOR):
            *_, product_name = self.aux_name.split(EXTRA_PRODUCT_SEPERATOR)
        else:
            product_name = self.data_source.aux_products[self.aux_name]["product"]
        return product_name

    @property
    def product_identifier(self):
        if self.aux_name is None:
            product_identifier = self.data_source.name
        else:
            product_identifier = self.aux_name
        return product_identifier

    @property
    def product_meta(self):
        if self.aux_name is None:
            product_meta = dict(name=self.product_identifier)
            if self.product_name == "user_function":
                if "input" not in self.data_source._meta:
                    raise Exception(
                        "To use a user-function on the primary data-set being"
                        " produced you need to define the channels used by defining"
                        " the section `input` in the input `meta.yaml` file"
                    )
                else:
                    product_meta["input"] = self.data_source._meta["input"]
        elif EXTRA_PRODUCT_SEPERATOR in self.aux_name:
            product_meta = {}
        else:
            product_meta = self.data_source.aux_products.get(self.aux_name)
            if product_meta is None:
                raise Exception(
                    f"Please define `{self.aux_name}` in the `aux_products`"
                    " group in meta.yaml"
                )

        product_meta["context"] = dict(
            datasource_path=self.data_path,
            product_identifier=self.product_identifier,
        )

        if "scene_mapping_strategy" not in product_meta:
            product_meta["scene_mapping_strategy"] = "single_scene_per_aux_time"

        return product_meta


class CheckForAuxiliaryFiles(luigi.Task, AuxTaskMixin):
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

        source_name = self.source_name
        product_name = self.product_name
        source_data_path = Path(self.data_path) / "source_data" / source_name

        try:
            tasks["product"] = build_query_tasks(
                source_name=source_name,
                product_name=product_name,
                source_data_path=source_data_path,
                time_intervals=self.data_source.time_intervals,
                product_meta=self.product_meta,
            )
        except IncompleteSourceDefition as ex:
            raise Exception(
                "There was a problem creating queries for your"
                f" definition of `{self.aux_name}`"
            ) from ex

        return tasks

    @property
    def dt_aux(self):
        dt_aux = self.product_meta.get("dt_aux", None)
        if dt_aux is not None:
            dt_aux = isodate.parse_duration(dt_aux)
        return dt_aux

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
        aux_scenes_by_time = self._filter_aux_times(
            create_scenes_from_input_queries(
                inputs=product_input,
                source_name=self.source_name,
                product=self.product_name,
            )
        )

        # match the aux files by their timestamp to the scene ids
        product_fn_for_scenes = _match_each_aux_time_to_scene_ids(
            aux_scenes_by_time=aux_scenes_by_time,
            scene_times=scene_times,
            scene_ids=scene_ids,
            dt_aux=self.dt_aux,
            strategy=self.product_meta["scene_mapping_strategy"],
        )

        self.output().write(product_fn_for_scenes)

    def _filter_aux_times(self, aux_scenes_by_time):
        # filter any files we don't want to use here

        file_exclusions_all = self.data_source._meta.get("file_exclusions", {})
        source_file_exclusions = file_exclusions_all.get(self.source_name, [])
        if len(source_file_exclusions) > 0:
            for dt in list(aux_scenes_by_time.keys()):
                scene_parts = aux_scenes_by_time[dt]
                if any(
                    filename in source_file_exclusions
                    for filename in scene_parts.values()
                ):
                    del aux_scenes_by_time[dt]

        return aux_scenes_by_time

    def output(self):
        data_source = self.data_source
        output = None
        if self.product_name == "user_function":
            product_name = f"{self.product_name}__{self.aux_name}"
        else:
            product_name = self.product_name

        if data_source.source == "goes16":
            p = Path(self.data_path) / "source_data" / self.source_name / product_name
            output = DBTarget(
                path=str(p), db_name="matched_scene_ids", db_type=data_source.db_type
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
