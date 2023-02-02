from pathlib import Path

from .derived_variables import compute_derived_variable  # noqa
from .pipeline import QueryForData  # noqa
from .query import parse_filename


def extract_variable(task_input, var_name):
    ds = task_input.open().rename(longitude="lon", latitude="lat")
    if var_name in ds.data_vars:
        da = ds[var_name]
    else:
        file_info = parse_filename(Path(task_input.path).name)
        ds["time"] = file_info["time"]
        da = compute_derived_variable(ds=ds, var_name=var_name)
    return da
