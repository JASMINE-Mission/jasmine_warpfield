#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Geocentric observer frame '''

from astropy.coordinates import FunctionTransformWithFiniteDifference
from astropy.coordinates import GCRS, ICRS
from astropy.coordinates import frame_transform_graph
from astropy.time import Time

from .base import Observer


__all__ = ['GeoCentric']


class GeoCentric(GCRS, Observer):
    ''' Geocentric observer frame aligned with the ICRS axes

    The observer is located at the geocenter by default. Astropy combines
    ``obsgeoloc`` and ``obsgeovel`` with the barycentric state of the Earth
    when transforming ICRS coordinates into this frame.
    '''

    def __init__(self, *args, **kwargs):
        if (
                len(args) == 1
                and isinstance(args[0], Time)
                and 'obstime' not in kwargs):
            kwargs['obstime'] = args[0]
            args = ()
        super().__init__(*args, **kwargs)


def _as_gcrs(frame, data=None):
    ''' Represent an observer frame as the corresponding GCRS frame '''
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
    ''' Transform ICRS coordinates using Astropy's GCRS implementation '''
    coordinate = icrs_coordinate.transform_to(_as_gcrs(geocentric_frame))
    return geocentric_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    GeoCentric,
    ICRS,
)
def _geocentric_to_icrs(geocentric_coordinate, icrs_frame):
    ''' Transform observer-centered coordinates back to ICRS '''
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
def _geocentric_to_geocentric(
        geocentric_coordinate, geocentric_frame):
    ''' Transform between geocentric frames with different attributes '''
    source = _as_gcrs(
        geocentric_coordinate,
        geocentric_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(geocentric_frame))
    return geocentric_frame.realize_frame(coordinate.data)
