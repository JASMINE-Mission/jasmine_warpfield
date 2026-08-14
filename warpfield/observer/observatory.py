#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Observer fixed to the rotating Earth"""

from astropy.coordinates import EarthLocation
from astropy.coordinates import FunctionTransformWithFiniteDifference
from astropy.coordinates import ICRS, QuantityAttribute
from astropy.coordinates import frame_transform_graph
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import numpy as np

from .geocentric import GeoCentric, _as_gcrs


__all__ = ['Observatory', 'ObservatoryN']


class Observatory(GeoCentric):
    """Observer fixed to a geodetic location on the rotating Earth

    Attributes:
        obstime: Time at which the observer state is evaluated.
        longitude: Geodetic longitude on the reference ellipsoid.
        latitude: Geodetic latitude on the reference ellipsoid.
        altitude: Height above the reference ellipsoid.
    """

    longitude = QuantityAttribute(
        default=None,
        unit=u.deg,
    )
    latitude = QuantityAttribute(
        default=None,
        unit=u.deg,
    )
    altitude = QuantityAttribute(
        default=None,
        unit=u.m,
    )

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], Time):
            names = ('obstime', 'longitude', 'latitude', 'altitude')
            if len(args) > len(names):
                raise TypeError(
                    'Observatory accepts at most four positional arguments.'
                )
            for name, value in zip(names[: len(args)], args, strict=True):
                if name in kwargs:
                    raise TypeError(f'`{name}` was specified more than once.')
                kwargs[name] = value
            args = ()

        missing = [
            name
            for name in (
                'obstime',
                'longitude',
                'latitude',
                'altitude',
            )
            if kwargs.get(name) is None
        ]
        if missing:
            raise TypeError(
                'Observatory requires '
                + ', '.join(f'`{name}`' for name in missing)
                + '.'
            )

        super().__init__(*args, **kwargs)

        longitude = np.asarray(self.longitude.to_value(u.deg))
        latitude = np.asarray(self.latitude.to_value(u.deg))
        altitude = np.asarray(self.altitude.to_value(u.m))
        if not np.all(np.isfinite(longitude)):
            raise ValueError('`longitude` should be finite.')
        if (
            not np.all(np.isfinite(latitude))
            or np.any(latitude < -90)
            or np.any(latitude > 90)
        ):
            raise ValueError(
                '`latitude` should be finite and in [-90, 90] deg.'
            )
        if not np.all(np.isfinite(altitude)):
            raise ValueError('`altitude` should be finite.')

    @property
    def earth_location(self):
        """Return the corresponding WGS84 Earth location"""
        return EarthLocation.from_geodetic(
            self.longitude,
            self.latitude,
            self.altitude,
        )

    @property
    def obsgeoloc(self):
        """Return the observatory position relative to the geocenter"""
        location, _ = self.earth_location.get_gcrs_posvel(self.obstime)
        return location

    @property
    def obsgeovel(self):
        """Return the observatory velocity relative to the geocenter"""
        _, velocity = self.earth_location.get_gcrs_posvel(self.obstime)
        return velocity

    @property
    def obsbaryloc(self):
        """Return the observatory position relative to the barycenter"""
        earth_location, _ = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return earth_location + self.obsgeoloc

    @property
    def obsbaryvel(self):
        """Return the observatory velocity relative to the barycenter"""
        _, earth_velocity = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return earth_velocity + self.obsgeovel


class ObservatoryN(Observatory):
    """Earth-fixed observer without annual aberration

    The suffix N denotes removal of the Earth's barycentric orbital velocity.
    Velocity caused by the Earth's rotation is retained.
    """

    @property
    def obsgeovel(self):
        """Return rotational velocity minus the Earth's orbital velocity"""
        local_velocity = super().obsgeovel
        _, earth_velocity = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return local_velocity - earth_velocity


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    Observatory,
)
def _icrs_to_observatory(icrs_coordinate, observatory_frame):
    """Transform ICRS coordinates to an Earth-fixed observatory"""
    coordinate = icrs_coordinate.transform_to(_as_gcrs(observatory_frame))
    return observatory_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    Observatory,
    ICRS,
)
def _observatory_to_icrs(observatory_coordinate, icrs_frame):
    """Transform observatory coordinates back to ICRS"""
    coordinate = _as_gcrs(
        observatory_coordinate,
        observatory_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    Observatory,
    Observatory,
)
def _observatory_to_observatory(observatory_coordinate, observatory_frame):
    """Transform between Earth-fixed observatory frames"""
    source = _as_gcrs(
        observatory_coordinate,
        observatory_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(observatory_frame))
    return observatory_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    ObservatoryN,
)
def _icrs_to_observatory_n(icrs_coordinate, observatory_frame):
    """Transform ICRS coordinates without annual aberration"""
    coordinate = icrs_coordinate.transform_to(_as_gcrs(observatory_frame))
    return observatory_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ObservatoryN,
    ICRS,
)
def _observatory_n_to_icrs(observatory_coordinate, icrs_frame):
    """Transform annual-aberration-free coordinates back to ICRS"""
    coordinate = _as_gcrs(
        observatory_coordinate,
        observatory_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ObservatoryN,
    ObservatoryN,
)
def _observatory_n_to_observatory_n(observatory_coordinate, observatory_frame):
    """Transform between annual-aberration-free observatory frames"""
    source = _as_gcrs(
        observatory_coordinate,
        observatory_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(observatory_frame))
    return observatory_frame.realize_frame(coordinate.data)
