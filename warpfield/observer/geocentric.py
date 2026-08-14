#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Geocentric observer frame"""

from astropy.coordinates import FunctionTransformWithFiniteDifference
from astropy.coordinates import BaseRADecFrame, CartesianRepresentation
from astropy.coordinates import GCRS, ICRS
from astropy.coordinates import frame_transform_graph
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u

from .base import Observer


__all__ = ['GeoCentric', 'GeoCentricN']


class GeoCentric(BaseRADecFrame, Observer):
    """Geocentric observer frame aligned with the ICRS axes

    The observer has zero position and velocity relative to the geocenter.
    Astropy combines this state with the barycentric state of the Earth when
    transforming ICRS coordinates into this frame.
    """

    def __init__(self, *args, **kwargs):
        if (
            len(args) == 1
            and isinstance(args[0], Time)
            and 'obstime' not in kwargs
        ):
            kwargs['obstime'] = args[0]
            args = ()
        super().__init__(*args, **kwargs)

    @property
    def obsgeoloc(self):
        """Return the observer location relative to the geocenter"""
        return CartesianRepresentation([0.0, 0.0, 0.0] * u.m)

    @property
    def obsgeovel(self):
        """Return the observer velocity relative to the geocenter"""
        return CartesianRepresentation([0.0, 0.0, 0.0] * u.m / u.s)


class GeoCentricN(GeoCentric):
    """Geocentric observer without annual aberration

    The suffix N denotes removal of the Earth's barycentric orbital velocity.
    The observer remains at the geocenter, so its total barycentric velocity
    is zero.
    """

    @property
    def obsgeovel(self):
        """Return the velocity that cancels the Earth's orbital motion"""
        _, earth_velocity = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return -earth_velocity


def _as_gcrs(frame, data=None):
    """Represent an observer frame as the corresponding GCRS frame"""
    attributes = {
        'obstime': frame.obstime,
        'obsgeoloc': frame.obsgeoloc,
        'obsgeovel': frame.obsgeovel,
    }
    if data is None:
        return GCRS(**attributes)
    return GCRS(data, **attributes)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    GeoCentric,
)
def _icrs_to_geocentric(icrs_coordinate, geocentric_frame):
    """Transform ICRS coordinates using Astropy's GCRS implementation"""
    coordinate = icrs_coordinate.transform_to(_as_gcrs(geocentric_frame))
    return geocentric_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GeoCentric,
    ICRS,
)
def _geocentric_to_icrs(geocentric_coordinate, icrs_frame):
    """Transform observer-centered coordinates back to ICRS"""
    coordinate = _as_gcrs(
        geocentric_coordinate,
        geocentric_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GeoCentric,
    GeoCentric,
)
def _geocentric_to_geocentric(geocentric_coordinate, geocentric_frame):
    """Transform between geocentric frames with different attributes"""
    source = _as_gcrs(
        geocentric_coordinate,
        geocentric_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(geocentric_frame))
    return geocentric_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    GeoCentricN,
)
def _icrs_to_geocentric_n(icrs_coordinate, geocentric_frame):
    """Transform ICRS coordinates without annual aberration"""
    coordinate = icrs_coordinate.transform_to(_as_gcrs(geocentric_frame))
    return geocentric_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GeoCentricN,
    ICRS,
)
def _geocentric_n_to_icrs(geocentric_coordinate, icrs_frame):
    """Transform annual-aberration-free coordinates back to ICRS"""
    coordinate = _as_gcrs(
        geocentric_coordinate,
        geocentric_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GeoCentricN,
    GeoCentricN,
)
def _geocentric_n_to_geocentric_n(geocentric_coordinate, geocentric_frame):
    """Transform between annual-aberration-free geocentric frames"""
    source = _as_gcrs(
        geocentric_coordinate,
        geocentric_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(geocentric_frame))
    return geocentric_frame.realize_frame(coordinate.data)
