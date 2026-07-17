#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Barycentrically stationary observer at the geocenter '''

from astropy.coordinates import FunctionTransformWithFiniteDifference
from astropy.coordinates import ICRS
from astropy.coordinates import frame_transform_graph
from astropy.coordinates import get_body_barycentric_posvel

from .geocentric import GeoCentric, _as_gcrs


__all__ = ['GeoCentricInertial']


class GeoCentricInertial(GeoCentric):
    ''' Observer at the geocenter with zero barycentric velocity

    ``obsgeovel`` cancels the barycentric velocity of the Earth.
    Consequently, the frame retains the geocentric position and parallax while
    suppressing aberration caused by the Earth's orbital velocity.
    '''

    @property
    def obsgeovel(self):
        ''' Return the velocity that cancels the Earth's orbital motion '''
        _, earth_velocity = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return -earth_velocity


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    GeoCentricInertial,
)
def _icrs_to_geocentric_inertial(icrs_coordinate, inertial_frame):
    ''' Transform ICRS coordinates using the configured GCRS state '''
    coordinate = icrs_coordinate.transform_to(_as_gcrs(inertial_frame))
    return inertial_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GeoCentricInertial,
    ICRS,
)
def _geocentric_inertial_to_icrs(inertial_coordinate, icrs_frame):
    ''' Transform geocentric-inertial coordinates back to ICRS '''
    coordinate = _as_gcrs(
        inertial_coordinate,
        inertial_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GeoCentricInertial,
    GeoCentricInertial,
)
def _geocentric_inertial_to_geocentric_inertial(
        inertial_coordinate, inertial_frame):
    ''' Transform between inertial frames with different attributes '''
    source = _as_gcrs(
        inertial_coordinate,
        inertial_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(inertial_frame))
    return inertial_frame.realize_frame(coordinate.data)
