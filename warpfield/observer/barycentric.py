#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Barycentric observer frame"""

from astropy.coordinates import FunctionTransformWithFiniteDifference
from astropy.coordinates import ICRS
from astropy.coordinates import frame_transform_graph
from astropy.time import Time

from .base import Observer


__all__ = ['BaryCentric']


class BaryCentric(ICRS, Observer):
    """Barycentric observer frame aligned with the ICRS axes

    This frame is an ICRS alias with an ``obstime`` attribute so that it can
    be used through the common Observer interface.
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


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    BaryCentric,
)
def _icrs_to_barycentric(icrs_coordinate, barycentric_frame):
    """Realize ICRS coordinates in the barycentric observer frame"""
    return barycentric_frame.realize_frame(icrs_coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    BaryCentric,
    ICRS,
)
def _barycentric_to_icrs(barycentric_coordinate, icrs_frame):
    """Realize barycentric observer coordinates in ICRS"""
    return icrs_frame.realize_frame(barycentric_coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    BaryCentric,
    BaryCentric,
)
def _barycentric_to_barycentric(barycentric_coordinate, barycentric_frame):
    """Transform between barycentric frames with different obstimes"""
    return barycentric_frame.realize_frame(barycentric_coordinate.data)
