import difflib
import itertools
import warnings

import numpy as np
import rioxarray as rxr
from rasterio.errors import NotGeoreferencedWarning


def edit_distance(s1, s2):
    """Computes the Levenshtein distance between two arrays (strings too).
    Such distance is the minimum number of operations needed to transform one array into
    the other, where an operation is an insertion, deletion, or substitution of a single
    item (like a char). This implementation (Wagner-Fischer algorithm with just 2 lines)
    uses O(min(|s1|, |s2|)) space.
    source: https://gist.github.com/p-hash/9e0f9904ce7947c133308fbe48fe032b
    editDistance([], [])
    0
    >>> editDistance([1, 2, 3], [2, 3, 5])
    2
    >>> tests = [["", ""], ["a", ""], ["", "a"], ["a", "a"], ["x", "a"],
    ...          ["aa", ""], ["", "aa"], ["aa", "aa"], ["ax", "aa"], ["a", "aa"], ["aa", "a"],
    ...          ["abcdef", ""], ["", "abcdef"], ["abcdef", "abcdef"],
    ...          ["vintner", "writers"], ["vintners", "writers"]];
    >>> [editDistance(s1, s2) for s1,s2 in tests]
    [0, 1, 1, 0, 1, 2, 2, 0, 1, 1, 1, 6, 6, 0, 5, 4]
    """
    if s1 == s2:
        return 0  # this is fast in Python
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    r1 = list(range(len(s2) + 1))
    r2 = [0] * len(r1)
    i = 0
    for c1 in s1:
        r2[0] = i + 1
        j = 0
        for c2 in s2:
            if c1 == c2:
                r2[j + 1] = r1[j]
            else:
                a1 = r2[j]
                a2 = r1[j]
                a3 = r1[j + 1]
                if a1 > a2:
                    if a2 > a3:
                        r2[j + 1] = 1 + a3
                    else:
                        r2[j + 1] = 1 + a2
                else:
                    if a1 > a3:
                        r2[j + 1] = 1 + a3
                    else:
                        r2[j + 1] = 1 + a1
            j += 1
        aux = r1
        r1 = r2
        r2 = aux
        i += 1
    return r1[-1]


# because it takes a while to open the HDF files we cache the variables in each
# HDF collection here. To update this list use
# `repr(list(_extract_and_rename_collection(hdf_collection, num))))`
COLLECTION_VARIABLES = [
    # collection 0
    [
        "solar_zenith_angle",
        "observed_all_sky_toa_sw_flux",
        "clear_sky_surface_sw_direct_flux",
        "clear_sky_surface_sw_diffuse_flux",
        "observed_all_sky_toa_lw_flux",
        "all_sky_surface_sw_direct_flux",
        "all_sky_surface_sw_diffuse_flux",
        "pristine_surface_sw_direct_flux",
        "pristine_surface_sw_diffuse_flux",
        "all_sky_noaerosol_surface_sw_direct_flux",
        "all_sky_noaerosol_surface_sw_diffuse_flux",
        "toa_uva_downwelling_flux",
        "toa_uvb_downwelling_flux",
        "all_sky_surface_uva_flux",
        "all_sky_surface_uvb_flux",
        "observed_all_sky_toa_wn_flux",
        "all_sky_surface_uv_index",
        "toa_par_downwelling_flux",
        "clear_sky_surface_par_direct_flux",
        "clear_sky_surface_par_diffuse_flux",
        "all_sky_surface_par_direct_flux",
        "all_sky_surface_par_diffuse_flux",
        "pristine_surface_par_direct_flux",
        "pristine_surface_par_diffuse_flux",
        "toa_outgoing_entropy_lw",
        "atmosphere_outgoing_entropy_lw",
        "observed_all_sky_toa_net_flux",
        "surface_outgoing_entropy_lw",
        "upward_surface_entropy_lw",
        "downward_surface_entropy_lw",
        "atmosphere_entropy_generation_by_lw_net",
        "surface_entropy_generation_by_lw_net",
        "toa_incoming_entropy_sw",
        "atmosphere_incoming_entropy_sw",
        "surface_incoming_entropy_sw",
        "atmosphere_entropy_generation_by_sw_net",
        "surface_entropy_generation_ by_sw_net",
        "observed_all_sky_toa_albedo",
        "number_of_ceres_sw_flux_observations",
        "number_of_ceres_lw_flux_observations",
        "number_of_geo_derived_sw_flux_observations",
        "number_ of_geo_derived_lw_flux_observations",
        "number_of_valid_initial_hourly_flux_computations",
        "number_of_valid_adjusted_hourly_flux_computations",
        "toa_sw_insolation",
        "surface_alti tude_above_sea_level",
        "ocean_percent_coverage",
        "initial_clear_sky_surface_sw_up_flux",
        "initial_clear_sky_surface_sw_down_flux",
        "initial_clear_sky_toa_sw_up_flux",
        "initial_c lear_sky_surface_lw_up_flux",
        "initial_clear_sky_surface_lw_down_flux",
        "initial_clear_sky_toa_lw_up_flux",
        "snow/ice_percent_coverage",
        "initial_all_sky_surface_sw_up_flux",
        "i nitial_all_sky_surface_sw_down_flux",
        "initial_all_sky_toa_sw_up_flux",
        "initial_all_sky_surface_lw_up_flux",
        "initial_all_sky_surface_lw_down_flux",
        "initial_all_sky_toa_lw_up_ flux",
        "initial_pristine_surface_sw_up_flux",
        "initial_pristine_surface_sw_down_flux",
        "initial_pristine_toa_sw_up_flux",
        "initial_pristine_surface_lw_up_flux",
        "observed_clear_ sky_toa_sw_flux",
        "initial_pristine_surface_lw_down_flux",
        "initial_pristine_toa_lw_up_flux",
        "initial_all_sky_noaerosol_surface_sw_up_flux",
        "initial_all_sky_noaerosol_surface_ sw_down_flux",
        "initial_all_sky_noaerosol_toa_sw_up_flux",
        "initial_all_sky_noaerosol_surface_lw_up_flux",
        "initial_all_sky_noaerosol_surface_lw_down_flux",
        "initial_all_sky_noa erosol_toa_lw_up_flux",
        "initial_all_sky_toa_satellite_emulated_wn_flux",
        "initial_clear_sky_toa_satellite_emulated_wn_flux",
        "observed_clear_sky_toa_lw_flux",
        "initial_precipit able_water",
        "initial_upper_tropospheric_relative_humidity",
        "initial_surface_albedo",
        "initial_skin_temperature",
        "initial_match_aerosol_optical_depth_at_0p55_um_band",
        "initia l_match_aerosol_optical_depth_at_0p84_um_band",
        "surface_pressure",
        "column_ozone",
        "observed_clear_sky_toa_wn_flux",
        "observed_clear_sky_toa_net_flux",
        "adjusted_all_sky_toa_sa tellite_emulated_wn_flux",
        "observed_clear_sky_toa_albedo",
        "adjusted_clear_sky_toa_satellite_emulated_wn_flux",
        "adjusted_precipitable_water",
        "adjusted_upper_tropospheric_rela tive_humidity",
        "adjusted_surface_albedo",
        "adjusted_skin_temperature",
        "adjusted_match_aerosol_optical_depth_at_0p55_um_band",
    ],
    # collection 1
    [
        "adjusted_cloud_ice_water_path",
        "adjusted_all_sky_toa_spectral_lw_up_flux",
        "adjusted_all_sky_surface_spectral_lw_up_flux",
        "adjusted_all_sky_surface_spectral_lw_down_flux",
        "observed_cloud_amount",
        "observed_cloud_visible_optical_depth_from_3p7_um_particle_size_retrieval",
        "observed_cloud_visible_optical_depth_linear_averaged,_from_3p7_um_particle_size_retrieval",
        "observed_cloud_infrared_emissivity",
        "observed_cloud_liquid_water_path_from_3p7_um_particle_size_retrieval",
        "observed_cloud_ice_water_path__from_3p7_um_particle_size_retrieval",
        "observed_cloud_top_pressure",
        "observed_cloud_top_temperature",
        "observed_cloud_top_height",
        "observed_cloud_effective_pressure",
        "observed_cloud_effective_temperature",
        "observed_cloud_effective_height",
        "observed_cloud_base_pressure",
        "observed_cloud_base_temperature",
        "observed_cloud_base_height",
        "observed_cloud_liquid_particle_radius_from_3p7_um_particle_size_retrieval",
        "observed_cloud_ice_particle_general_effective_radius_from_3p7_um_particle_size_retrieval",
        "observed_cloud_particle_phase_from_3p7_um_particle_size_retrieval",
        "initial_cloud_amount",
        "initial_cloud_temperature",
        "initial_cloud_optical_depth",
        "initial_cloud_liquid_water_path",
        "initial_cloud_ice_water_path",
        "adjusted_cloud_amount",
        "adjusted_cloud_temperature",
        "adjusted_cloud_optical_depth",
        "adjusted_cloud_liquid_water_path",
    ],
    # collection 2
    [
        "adjusted_all_sky_toa_spectral_sw_down_flux",
        "adjusted_all_sky_toa_spectral_sw_up_flux",
        "adjusted_all_sky_surface_spectral_sw_down_flux",
        "adjusted_all_sky_surface_spectral_sw_up_flux",
    ],
    # collection 3
    [
        "adjusted_clear_sky_sw_up_flux",
        "adjusted_clear_sky_sw_down_flux",
        "adjusted_clear_sky_lw_up_flux",
        "adjusted_clear_sky_lw_down_flux",
        "adjusted_all_sky_sw_up_flux",
        "adjusted_all_sky_sw_down_flux",
        "adjusted_all_sky_lw_up_flux",
        "adjusted_all_sky_lw_down_flux",
        "adjusted_pristine_sw_up_flux",
        "adjusted_pristine_sw_down_flux",
        "adjusted_pristine_lw_up_flux",
        "adjusted_pristine_lw_down_flux",
        "adjusted_all_sky_noaerosol_sw_up_flux",
        "adjusted_all_sky_noaerosol_sw_down_flux",
        "adjusted_all_sky_noaerosol_lw_up_flux",
        "adjusted_all_sky_noaerosol_lw_down_flux",
    ],
]

ALL_KNOWN_VARS = list(itertools.chain(*COLLECTION_VARIABLES))


def _extract_and_rename_collection(hdf_collection, num):
    ds = hdf_collection[num]

    # make some more sensible names
    new_names = {
        v: ds[v]
        .long_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "p")
        for v in ds.data_vars
    }

    return ds.rename(new_names)


def _cleanup_and_extract_var(ds, var_name):
    # add lat/lon grid
    ds["lat"] = 90 - ds.y
    ds["lon"] = ds.x - 180
    ds = ds.swap_dims(dict(x="lon", y="lat"))

    da = ds[var_name]

    # add time coord
    da["time"] = ("band",), [
        np.datetime64(da.RANGEBEGINNINGDATE) + np.timedelta64(n, "h")
        for n in range(int(da.band.count()))
    ]
    da = da.swap_dims(dict(band="time"))
    da = da.drop(["band", "spatial_ref"])

    return da


def _extract_scene_var_from_hdf_file(hdf_collection, var_name, timestamp):
    ds = None
    for collection_num, collection_variables in enumerate(COLLECTION_VARIABLES):
        if var_name in collection_variables:
            if collection_num != 0:
                raise NotImplementedError(
                    "Not yet worked out how to extract variables from other than collection 1"
                )
            ds = _extract_and_rename_collection(
                hdf_collection=hdf_collection, num=collection_num
            )
            break

    if ds is None:
        raise NotImplementedError(
            f"{var_name} isn't available in syn1deg data. "
            f"Known variables: {', '.join(ALL_KNOWN_VARS)}"
        )

    da = _cleanup_and_extract_var(ds=ds, var_name=var_name)

    # NB: scene timestamp might not be exactly the same as the time of this
    # dataset, but we're assuming here that we're selecting from a set of
    # timestamps that are close enough (that these scene has been associated
    # with this file previously)
    da_scene = da.sel(time=timestamp, method="nearest")

    return da_scene


def extract_variable(task_input, var_name, timestamp):
    with warnings.catch_warnings():
        fn_modis = task_input.path
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        hdf_collection = rxr.open_rasterio(fn_modis)

    def _extract_var(v):
        return _extract_scene_var_from_hdf_file(
            hdf_collection=hdf_collection, var_name=v, timestamp=timestamp
        )

    if var_name in ALL_KNOWN_VARS:
        da = _extract_var(v=var_name)
    elif var_name == "toa_sw_cre":
        da = calc_toa_sw_cre(
            da_observed_clear_sky_toa_albedo=_extract_var(v="clear_sky_toa_albedo"),
            da_observed_all_sky_toa_albedo=_extract_var(v="all_sky_toa_albedo"),
            da_toa_sw_insolation=_extract_var(v="toa_sw_insolation"),
        )
    elif var_name == "toa_lw_cre":
        da = calc_toa_lw_cre(
            da_observed_clear_sky_toa_lw_flux=_extract_var(v="clear_sky_toa_lw_flux"),
            da_observed_all_sky_toa_lw_flux=_extract_var(v="all_sky_toa_lw_flux"),
        )
    elif var_name == "toa_net_cre":
        da_toa_sw_cre = calc_toa_sw_cre(
            da_observed_clear_sky_toa_albedo=_extract_var(v="clear_sky_toa_albedo"),
            da_observed_all_sky_toa_albedo=_extract_var(v="all_sky_toa_albedo"),
            da_toa_sw_insolation=_extract_var(v="toa_sw_insolation"),
        )
        da_toa_lw_cre = calc_toa_lw_cre(
            da_observed_clear_sky_toa_lw_flux=_extract_var(v="clear_sky_toa_lw_flux"),
            da_observed_all_sky_toa_lw_flux=_extract_var(v="all_sky_toa_lw_flux"),
        )
        da = da_toa_sw_cre + da_toa_lw_cre
        da.attrs["units"] = da_toa_lw_cre.units
        da.attrs["long_name"] = "TOA NET CRE"
    else:
        most_similar_vars = difflib.get_close_matches(
            var_name, ALL_KNOWN_VARS, n=10, cutoff=0.2
        )

        raise Exception(
            f"Variable `{var_name}` not found in MODIS SYN1Deg dataset. "
            f"Available variables are: {', '.join(ALL_KNOWN_VARS)}. "
            f"The most similar to the variable you requested are: {', '.join(most_similar_vars)}"
        )

    da = da.sortby("lat")

    extra_attrs = ["INPUTPOINTER"]
    for attr in extra_attrs:
        if attr in da.attrs:
            del da.attrs[attr]

    return da


def calc_toa_sw_cre(
    da_observed_clear_sky_toa_albedo,
    da_observed_all_sky_toa_albedo,
    da_toa_sw_insolation,
):
    da_toa_sw_cre = (
        da_observed_clear_sky_toa_albedo - da_observed_all_sky_toa_albedo
    ) * da_toa_sw_insolation
    da_toa_sw_cre.attrs["units"] = da_toa_sw_insolation.units
    da_toa_sw_cre.attrs["long_name"] = "TOA SW CRE"
    da_toa_sw_cre.name = "toa_sw_cre"
    return da_toa_sw_cre


def calc_toa_lw_cre(da_observed_all_sky_toa_lw_flux, da_observed_clear_sky_toa_lw_flux):
    da_toa_lw_cre = -(
        da_observed_all_sky_toa_lw_flux - da_observed_clear_sky_toa_lw_flux
    )
    da_toa_lw_cre.attrs["units"] = da_observed_clear_sky_toa_lw_flux.units
    da_toa_lw_cre.attrs["long_name"] = "TOA LW CRE"
    da_toa_lw_cre.name = "toa_lw_cre"
    return da_toa_lw_cre
