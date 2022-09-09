import functools

import numpy as np


def _filter_by_percentile(ds, var_name, frac=90.0, part="upper"):
    if part == "lower":
        q = frac
    elif part == "upper":
        q = 100.0 - frac
    else:
        raise Exception(f"Invalid percentile part `{part}`")

    v = ds.dropna(dim="object_id")[var_name]
    vlim = np.percentile(v, q=q)

    if part == "lower":
        return ds.where(ds[var_name] < vlim, drop=True)

    return ds.where(ds[var_name] > vlim, drop=True)


def make_filter_fn(prop_name, op_name, s_value):
    if op_name in ["upper_percentile", "lower_percentile"]:
        value = float(s_value)
        part, _ = op_name.split("_")
        fn = functools.partial(
            _filter_by_percentile, frac=value, part=part, var_name=prop_name
        )
    if op_name == "isnan":
        if s_value == "True":
            op_fn = np.isnan
        elif s_value == "False":
            op_fn = lambda v: np.logical_not(np.isnan(v))  # noqa
        else:
            raise NotImplementedError(s_value)

        fn = lambda ds: ds.where(op_fn(ds[prop_name]), drop=True)  # noqa
    else:
        op = dict(
            lt="less_than",
            gt="greater_than",
            eq="equal",
            lte="less_equal",
            gte="greater_equal",
            isnan="isnan",
        )[op_name]
        op_fn = getattr(np, op.replace("_than", ""))
        value = float(s_value)

        fn = lambda ds: ds.where(
            op_fn(getattr(ds, prop_name), value), drop=True
        )  # noqa

    return fn


def _defs_iterator(filter_defs):
    s_filters = filter_defs.split(",")
    for s_filter in s_filters:
        try:
            s_prop_and_op, s_value = s_filter.split("=")
            i = s_prop_and_op.rfind("__")
            if i == -1:
                prop_name = s_prop_and_op
                op_name = "eq"
            else:
                prop_name, op_name = s_prop_and_op[:i], s_prop_and_op[i + 2 :]
            yield (prop_name, op_name, s_value)
        except (IndexError, ValueError) as ex:
            raise Exception(f"Malformed filter definition: `{s_filter}`") from ex


def parse_defs(filter_defs):
    filters = []

    for f_def in _defs_iterator(filter_defs):
        filt = dict(fn=None, reqd_props=[], extra=[])
        prop_name, op_name, s_value = f_def
        fn = make_filter_fn(prop_name, op_name, s_value)
        filt["reqd_props"].append(prop_name)
        filt["fn"] = fn
        filters.append(filt)

    return filters
