from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from convml_tt.data.sources.pipeline import (
    GenerateRegriddedScenes,
    GenerateSceneIDs,
    parse_scene_id,
)
from tqdm import tqdm


def collect_processed_scene_ids(task, data_path):
    try:
        output = task.output().open()
        scene_ids = list(output.keys())
    except AttributeError:
        output = task.output()
        scene_ids = list(output.keys())
        pass

    def _data_exists(s_id):
        resource = output[s_id]
        if isinstance(resource, list):
            return all(
                [
                    (Path(data_path) / "source_data/goes16/" / o).exists()
                    for o in resource
                ]
            )
        elif "data" in resource:
            return resource["data"].exists()
        else:
            raise NotImplementedError(resource)

    task_name = task.__class__.__name__
    scene_ids = [s_id for s_id in tqdm(scene_ids, desc=task_name) if _data_exists(s_id)]

    times = [parse_scene_id(scene_id)[1] for scene_id in scene_ids]
    ds = xr.Dataset(dict(time=sorted(times)))
    if len(times) == 0:
        print(f"no data for {task_name}")
        return None

    name = task.__class__.__name__
    ds["completed_subtasks"] = ("time"), np.ones(len(ds.time))
    ds["task"] = name

    return ds


def main(data_path=".", time_interval="1D"):
    tasks = [
        GenerateSceneIDs(data_path=data_path),
        GenerateRegriddedScenes(data_path=data_path),
    ]

    datasets = []
    for task in tasks:
        ds = collect_processed_scene_ids(task, data_path=data_path)
        if ds is not None:
            datasets.append(ds)
    ds = xr.concat(datasets, dim="task")

    fig, ax = plt.subplots(figsize=(12, 4))
    da = ds.resample(time=time_interval).count().completed_subtasks
    da.plot(ax=ax, marker=".", hue="task")
    ax.set_ylim(0, 1.2 * da.max())

    fn = "processed_scenes.png"
    fig.savefig(fn, bbox_inches="tight")
    print(f"Saved plot of processed scenes statistics to `{fn}`")


if __name__ == "__main__":
    main()
