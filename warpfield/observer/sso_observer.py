#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Observer in a circular Sun-synchronous Earth orbit"""

from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import FunctionTransformWithFiniteDifference
from astropy.coordinates import ICRS, Longitude, QuantityAttribute
from astropy.coordinates import frame_transform_graph
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.constants as c
import astropy.units as u
import numpy as np

from .geocentric import GeoCentric, _as_gcrs


__all__ = ['SSOObserver', 'SSOObserverN']


_EARTH_J2 = 1.08262668e-3
_TROPICAL_YEAR = 365.2421897 * u.day


class SSOObserver(GeoCentric):
    """Observer in a circular Sun-synchronous orbit

    Attributes:
        obstime: Time at which the orbital state is evaluated.
        phase: Orbital phase, where zero is the ascending node.
        altitude: Circular-orbit altitude above the nominal Earth radius.
        ltan: Mean local solar time of the ascending node.
    """

    phase = QuantityAttribute(
        default=0.0,
        unit=u.one,
    )
    altitude = QuantityAttribute(
        default=600.0 * u.km,
        unit=u.km,
    )
    ltan = QuantityAttribute(
        default=6.0 * u.hourangle,
        unit=u.hourangle,
    )

    def __init__(self, *args, **kwargs):
        if (
            len(args) == 1
            and isinstance(args[0], Time)
            and 'obstime' not in kwargs
        ):
            kwargs['obstime'] = args[0]
            args = ()
        super().__init__(*args, **kwargs)

        altitude = np.asarray(self.altitude.to_value(u.km))
        if not np.all(np.isfinite(altitude)) or np.any(altitude <= 0):
            raise ValueError('`altitude` should be finite and positive.')

        phase = np.asarray(self.phase.to_value(u.one))
        if not np.all(np.isfinite(phase)):
            raise ValueError('`phase` should be finite.')

        ltan = np.asarray(self.ltan.to_value(u.hourangle))
        if (
            not np.all(np.isfinite(ltan))
            or np.any(ltan < 0)
            or np.any(ltan >= 24)
        ):
            raise ValueError('`ltan` should be in the range [0, 24) h.')

        # Evaluate once to reject altitudes for which the first-order J2
        # condition has no circular Sun-synchronous solution.
        self.inclination

    @property
    def orbital_radius(self):
        """Return the circular-orbit radius"""
        return c.R_earth + self.altitude

    @property
    def orbital_velocity(self):
        """Return the circular-orbit speed"""
        return np.sqrt(c.GM_earth / self.orbital_radius)

    @property
    def orbital_period(self):
        """Return the circular-orbit period"""
        return 2 * np.pi * np.sqrt(self.orbital_radius**3 / c.GM_earth)

    @property
    def inclination(self):
        """Return the retrograde inclination satisfying the J2 condition"""
        radius = self.orbital_radius
        mean_motion = np.sqrt(c.GM_earth / radius**3)
        solar_rate = (2 * np.pi * u.rad / _TROPICAL_YEAR).to(
            1 / u.s, equivalencies=u.dimensionless_angles()
        )
        cosine = (
            (
                -2
                * solar_rate
                / (3 * _EARTH_J2 * mean_motion * (c.R_earth / radius) ** 2)
            )
            .decompose()
            .value
        )
        if np.any(np.abs(cosine) > 1):
            raise ValueError(
                '`altitude` does not admit a circular '
                'Sun-synchronous solution.'
            )
        return np.arccos(cosine) * u.rad

    @property
    def mean_sun_right_ascension(self):
        """Return the right ascension of the fictitious mean Sun"""
        gmst = self.obstime.sidereal_time('mean', 'greenwich')
        ut1_fraction = np.mod(self.obstime.ut1.jd + 0.5, 1.0)
        mean_solar_hour_angle = (24 * ut1_fraction - 12) * u.hourangle
        return Longitude(gmst - mean_solar_hour_angle)

    @property
    def raan(self):
        """Return the right ascension of the ascending node"""
        offset = (self.ltan - 12 * u.hourangle).to(u.deg)
        return Longitude(self.mean_sun_right_ascension + offset)

    @property
    def obsgeoloc(self):
        """Return the satellite position relative to the geocenter"""
        raan = self.raan.to_value(u.rad)
        inclination = self.inclination.to_value(u.rad)
        argument = 2 * np.pi * self.phase.to_value(u.one)

        ascending = np.stack([
            np.cos(raan),
            np.sin(raan),
            np.zeros_like(raan),
        ])
        transverse = np.stack([
            -np.sin(raan) * np.cos(inclination),
            +np.cos(raan) * np.cos(inclination),
            +np.sin(inclination) * np.ones_like(raan),
        ])
        direction = (
            np.cos(argument) * ascending + np.sin(argument) * transverse
        )
        return CartesianRepresentation(self.orbital_radius * direction)

    @property
    def obsgeovel(self):
        """Return the satellite velocity relative to the geocenter"""
        raan = self.raan.to_value(u.rad)
        inclination = self.inclination.to_value(u.rad)
        argument = 2 * np.pi * self.phase.to_value(u.one)

        ascending = np.stack([
            np.cos(raan),
            np.sin(raan),
            np.zeros_like(raan),
        ])
        transverse = np.stack([
            -np.sin(raan) * np.cos(inclination),
            +np.cos(raan) * np.cos(inclination),
            +np.sin(inclination) * np.ones_like(raan),
        ])
        direction = (
            -np.sin(argument) * ascending + np.cos(argument) * transverse
        )
        return CartesianRepresentation(self.orbital_velocity * direction)

    @property
    def obsbaryloc(self):
        """Return the satellite position relative to the barycenter"""
        earth_location, _ = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return earth_location + self.obsgeoloc

    @property
    def obsbaryvel(self):
        """Return the satellite velocity relative to the barycenter"""
        _, earth_velocity = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return earth_velocity + self.obsgeovel


class SSOObserverN(SSOObserver):
    """Sun-synchronous observer without annual aberration

    The suffix N denotes removal of the Earth's barycentric orbital velocity.
    The satellite's velocity relative to the geocenter is retained.
    """

    @property
    def obsgeovel(self):
        """Return orbital velocity minus the Earth's orbital velocity"""
        local_velocity = super().obsgeovel
        _, earth_velocity = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return local_velocity - earth_velocity


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    SSOObserver,
)
def _icrs_to_sso_observer(icrs_coordinate, observer_frame):
    """Transform ICRS coordinates using the SSO observer state"""
    coordinate = icrs_coordinate.transform_to(_as_gcrs(observer_frame))
    return observer_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    SSOObserver,
    ICRS,
)
def _sso_observer_to_icrs(observer_coordinate, icrs_frame):
    """Transform SSO observer coordinates back to ICRS"""
    coordinate = _as_gcrs(
        observer_coordinate,
        observer_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    SSOObserver,
    SSOObserver,
)
def _sso_observer_to_sso_observer(observer_coordinate, observer_frame):
    """Transform between SSO observer frames"""
    source = _as_gcrs(
        observer_coordinate,
        observer_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(observer_frame))
    return observer_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    SSOObserverN,
)
def _icrs_to_sso_observer_n(icrs_coordinate, observer_frame):
    """Transform ICRS coordinates without annual aberration"""
    coordinate = icrs_coordinate.transform_to(_as_gcrs(observer_frame))
    return observer_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    SSOObserverN,
    ICRS,
)
def _sso_observer_n_to_icrs(observer_coordinate, icrs_frame):
    """Transform annual-aberration-free SSO coordinates back to ICRS"""
    coordinate = _as_gcrs(
        observer_coordinate,
        observer_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    SSOObserverN,
    SSOObserverN,
)
def _sso_observer_n_to_sso_observer_n(observer_coordinate, observer_frame):
    """Transform between annual-aberration-free SSO frames"""
    source = _as_gcrs(
        observer_coordinate,
        observer_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(observer_frame))
    return observer_frame.realize_frame(coordinate.data)
