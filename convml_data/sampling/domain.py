"""
Utilities to create (approximate) square tiles from lat/lon satelite data
"""
import cartopy.crs as ccrs
import numpy as np
import regridcart as rc
import xarray as xr


class LocalCartesianSquareTileDomain(rc.LocalCartesianDomain):
    def __init__(self, central_latitude, central_longitude, size, x_c=0.0, y_c=0.0):
        """
        Create a locally Cartesian square tile with `size` (in meters)
        """
        self.size = size
        super().__init__(
            central_latitude=central_latitude,
            central_longitude=central_longitude,
            l_meridional=size,
            l_zonal=size,
            x_c=x_c,
            y_c=y_c,
        )

    def get_grid(self, N):
        dx = self.size / N
        ds_grid = super().get_grid(dx=dx)
        # the floating point devision in `get_grid` when we're passing in `dx`
        # means that sometimes we might produce an extra set of x- or y-values,
        # so we do an index-selection here to ensure those aren't included
        return ds_grid.isel(x=slice(0, N), y=slice(0, N))


class CartesianSquareTileDomain(rc.CartesianDomain):
    def __init__(self, x_c, y_c, size):
        """
        Create a Cartesian square tile with `size` (in meters) centered at a
        (x,y)=(x_c,y_c) location
        """
        self.size = size
        super().__init__(x_c=x_c, y_c=y_c, l_meridional=size, l_zonal=size)

    def get_grid(self, N):
        dx = self.size / N
        ds_grid = super().get_grid(dx=dx)
        # the floating point devision in `get_grid` when we're passing in `dx`
        # means that sometimes we might produce an extra set of x- or y-values,
        # so we do an index-selection here to ensure those aren't included
        return ds_grid.isel(x=slice(0, N), y=slice(0, N))

    def locate_in_latlon_domain(self, domain):
        """
        Create a new local cartesian domain centered on latlon coordinates
        relative to the origin in the parent domain
        """
        tile_latlon = domain.latlon_from_xy(
            x=domain.x_c - self.x_c, y=domain.y_c - self.y_c
        )
        return LocalCartesianSquareTileDomain(
            central_latitude=float(tile_latlon[1]),
            central_longitude=float(tile_latlon[0]),
            size=self.size,
            x_c=self.x_c,
            y_c=self.y_c,
        )


class SourceDataDomain:
    """
    Represents that the domain information should be extracted from a source dataset
    """

    def generate_from_dataset(self, ds):
        """
        Create an actual domain instance from the provided dataset. This will
        be the largest possible domain that can fit.
        """
        if "x" in ds.coords and "y" in ds.coords:
            x_min, x_max = ds.x.min().data, ds.x.max().data
            y_min, y_max = ds.y.min().data, ds.y.max().data

            l_zonal = x_max - x_min
            l_meridinonal = y_max - y_min
            x_c = 0.5 * (x_min + x_max)
            y_c = 0.5 * (y_min + y_max)
            return rc.CartesianDomain(
                l_meridional=l_meridinonal, l_zonal=l_zonal, x_c=x_c, y_c=y_c
            )
        elif "lat" in ds.coords and "lon" in ds.coords:
            lat_min, lat_max = ds.lat.min(), ds.lat.max()
            lon_min, lon_max = ds.lon.min(), ds.lon.max()

            # TODO: calculating the centre like this is a bit crude since on a
            # curve surface the centreline also curves and so on the northern
            # hemisphere we may want a domain slightly further north to fit it
            # in
            lat_center = float((lat_min + lat_max) / 2.0)
            lon_center = float((lon_min + lon_max) / 2.0)
            # create a domain with no span but positioned at the correct point
            # so that ew can calculate the maximum zonal and meridional span
            # from the lat/lon of the corners
            dummy_domain = rc.LocalCartesianDomain(
                central_latitude=lat_center,
                central_longitude=lon_center,
                l_zonal=0.0,
                l_meridional=0.0,
            )

            edges = dict(
                W=("lon", 0, np.max),
                E=("lon", -1, np.min),
                S=("lat", 0, np.max),
                N=("lat", -1, np.min),
            )

            edge_positions = {}
            for edge, (dim, idx, op) in edges.items():
                ds_edge = ds.isel({dim: idx})
                lat_edge = ds_edge.lat
                lon_edge = ds_edge.lon

                lats_edge, lons_edge = xr.broadcast(lat_edge, lon_edge)

                # find xy-position of the edge
                xy_edge = dummy_domain.crs.transform_points(
                    x=lons_edge.data,
                    y=lats_edge.data,
                    z=np.zeros_like(lats_edge.data),
                    src_crs=ccrs.PlateCarree(),
                )

                if dim == "lon":
                    posn = op(xy_edge[..., 0])
                elif dim == "lat":
                    posn = op(xy_edge[..., 1])
                else:
                    raise NotImplementedError

                edge_positions[edge] = posn

            lx = edge_positions["E"] - edge_positions["W"]
            ly = edge_positions["N"] - edge_positions["S"]

            # round down to nearest meter
            lx = np.round(lx, decimals=0)
            ly = np.round(ly, decimals=0)

            # TODO: remove crop by calculating centre offset which still fit
            # rectangular domain
            crop = 0.95

            # finally we return a domain with the correct span
            return rc.LocalCartesianDomain(
                central_latitude=lat_center,
                central_longitude=lon_center,
                l_zonal=crop * lx,
                l_meridional=crop * ly,
            )
        else:
            raise NotImplementedError(ds.coords)


def _calc_latlon_center(lat, lon):
    """
    Calculate lat/lon center for a set of points given by lat/lon coordinates.
    Units are expected to be in degrees

    based on: https://stackoverflow.com/q/6671183/271776
    """
    lat = lat * np.pi / 180.0
    lon = lon * np.pi / 180.0

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    # compute average x, y and z coordinates.
    x_ = x.mean()
    y_ = y.mean()
    z_ = z.mean()

    # convert average x, y, z coordinate to latitude and longitude
    lon_center = np.arctan2(y_, x_)
    hyp_center = np.sqrt(x_ * x_ + y_ * y_)
    lat_center = np.arctan2(z_, hyp_center)

    print(lat_center, lon_center)

    return lat_center * 180.0 / np.pi, lon_center * 180.0 / np.pi


class LatLonPointsSpanningDomain(rc.LocalCartesianDomain):
    def __init__(self, da_lat, da_lon, padding=1.1):
        lat_origin, lon_origin = _calc_latlon_center(
            lat=da_lat.values, lon=da_lon.values
        )

        # create a domain with no span but positioned at the correct point
        # so that ew can calculate the maximum zonal and meridional span
        # from the lat/lon of the corners
        dummy_domain = rc.LocalCartesianDomain(
            central_latitude=lat_origin,
            central_longitude=lon_origin,
            l_zonal=0.0,
            l_meridional=0.0,
        )

        # find xy-position of all lat/lon positions in trajectories
        xy_pts = dummy_domain.crs.transform_points(
            x=da_lon,
            y=da_lat,
            z=np.zeros_like(da_lon.data),
            src_crs=ccrs.PlateCarree(),
        )

        x_pts = xy_pts[..., 0]
        y_pts = xy_pts[..., 1]

        l_zonal = padding * 2.0 * max([abs(x_pts.min()), x_pts.max()])
        l_meridional = padding * 2.0 * max([abs(y_pts.min()), y_pts.max()])

        super().__init__(
            central_latitude=lat_origin,
            central_longitude=lon_origin,
            l_zonal=l_zonal,
            l_meridional=l_meridional,
        )


class TrajectoriesSpanningDomain(LatLonPointsSpanningDomain):
    def __init__(self, ds_trajectories, padding=1.2):
        self.ds = ds_trajectories
        da_lon = self.ds.lon
        da_lat = self.ds.lat
        super().__init__(da_lat=da_lat, da_lon=da_lon, padding=padding)

    def plot_trajectories(self, ax=None, ds_traj=None, **kwargs):
        if ax is None:
            ax = self.plot_outline()

        if ds_traj is None:
            ds_traj = self.ds

        return ax.plot(ds_traj.lon, ds_traj.lat, transform=ccrs.PlateCarree(), **kwargs)
