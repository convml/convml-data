# from genesis codebase
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import xarray as xr


def _make_equally_spaced_bins(x, dx, n_pad=2):
    def _get_finite_range(vals):
        # turns infs into nans and then we can uses nanmax nanmin
        v = vals.where(~np.isinf(vals), np.nan)
        return np.nanmin(v), np.nanmax(v)

    x_min, x_max = _get_finite_range(x)
    bins = np.arange(
        (math.floor(x_min / dx) - n_pad) * dx,
        (math.ceil(x_max / dx) + n_pad) * dx + dx,
        dx,
    )

    return bins


def scalar_binning_2d(
    ds, x, y, v, dx, dy, statistic="mean", drop_nan_and_inf=False, min_points=None
):
    """
    Compute the 2D binning of the scalar `v` with the
    variables `x` and `y` with discretisation `dx`, `dy` so that the integral
    over `x` and `y` (with increments `dx` and `dy`) equals the sum of `v`
    """
    # we can't bin on values which are either nan or inf, so we filter those out here
    def should_mask(x_):
        return np.logical_or(np.isnan(x_), np.isinf(x_))

    values_mask = np.logical_or(
        should_mask(ds[x]), np.logical_or(should_mask(ds[y]), should_mask(ds[v]))
    )

    if drop_nan_and_inf:
        ds_ = ds.where(~values_mask, drop=True)
    else:
        if np.any(values_mask):
            raise Exception(
                "There are some inf or nan values, but we can't bin on these."
                " Set `drop_nan_and_inf=True` to filter these out"
            )
        else:
            ds_ = ds

    x_bins = _make_equally_spaced_bins(ds_[x], dx=dx)
    y_bins = _make_equally_spaced_bins(ds_[y], dx=dy)

    da_x = ds_[x]
    da_y = ds_[y]
    da_v = ds_[v]

    def center(x_):
        return 0.5 * (x_[1:] + x_[:-1])

    x_ = center(x_bins)
    y_ = center(y_bins)

    # bin the values to compute a distribution of the scalar values `v`
    bins = (x_bins, y_bins)
    if statistic == "sem":
        statistic = scipy.stats.sem
    arr = scipy.stats.binned_statistic_2d(
        da_x, da_y, da_v, statistic=statistic, bins=bins
    ).statistic

    if min_points:
        arr_count = scipy.stats.binned_statistic_2d(
            da_x, da_y, da_v, statistic="count", bins=bins
        ).statistic

        arr = np.where(arr_count > min_points, arr, np.nan)

    da_x_sd = xr.DataArray(x_, attrs=da_x.attrs, dims=(da_x.name), name=da_x.name)
    da_y_sd = xr.DataArray(y_, attrs=da_y.attrs, dims=(da_y.name), name=da_y.name)

    long_name = da_v.long_name.replace("per object", "")
    units = f"{da_v.units} / ({da_x.units} {da_y.units})"
    da_sd = xr.DataArray(
        arr.T,
        dims=(da_y.name, da_x.name),
        coords={da_x.name: da_x_sd, da_y.name: da_y_sd},
        attrs=dict(long_name=long_name, units=units),
    )
    return da_sd


def dist_plots(
    ds, x, y, v, dx, dy, statistics=["mean", "sem"], cmap="jet", min_points=50
):
    col_max = 2
    N_subplots = len(statistics)
    if N_subplots > col_max:
        N_cols = col_max
    N_rows = math.ceil(N_subplots / N_cols)

    fig, axes = plt.subplots(
        ncols=N_cols, nrows=N_rows, figsize=(6 * N_cols, 4 * N_rows)
    )

    axes = np.atleast_1d(axes).flatten()
    for ax, statistic in zip(axes, statistics):
        da_binned = scalar_binning_2d(
            ds=ds,
            x=x,
            y=y,
            v=v,
            dx=dx,
            dy=dy,
            drop_nan_and_inf=True,
            statistic=statistic,
            min_points=min_points,
        )
        da_binned.plot(ax=ax, cmap=cmap)
        ax.set_aspect(1.0)
        if statistic == "sem":
            ax.set_title("standard error in mean")
        else:
            ax.set_title(statistic)

    fig.tight_layout()
    return fig, axes
