#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Observer with an arbitrary state in BCRS '''

from astropy.coordinates import BaseRADecFrame
from astropy.coordinates import CartesianRepresentationAttribute
from astropy.coordinates import FunctionTransformWithFiniteDifference
from astropy.coordinates import ICRS
from astropy.coordinates import frame_transform_graph
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u

from .base import Observer
from .geocentric import _as_gcrs


__all__ = ['BCRSObserver']


class BCRSObserver(BaseRADecFrame, Observer):
    ''' Observer with an arbitrary barycentric position and velocity

    Attributes:
        obstime: Time at which the observer state is defined.
        obsbaryloc: Observer position relative to the Solar System barycenter.
        obsbaryvel: Observer velocity relative to the Solar System barycenter.
    '''

    obsbaryloc = CartesianRepresentationAttribute(
        default=None,
        unit=u.m,
    )
    obsbaryvel = CartesianRepresentationAttribute(
        default=None,
        unit=u.m / u.s,
    )

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], Time):
            names = ('obstime', 'obsbaryloc', 'obsbaryvel')
            if len(args) > len(names):
                raise TypeError(
                    'BCRSObserver accepts at most three positional '
                    'arguments.')
            for name, value in zip(
                    names[:len(args)], args, strict=True):
                if name in kwargs:
                    raise TypeError(
                        f'`{name}` was specified more than once.')
                kwargs[name] = value
            args = ()

        missing = [
            name for name in ('obstime', 'obsbaryloc', 'obsbaryvel')
            if kwargs.get(name) is None
        ]
        if missing:
            raise TypeError(
                'BCRSObserver requires '
                + ', '.join(f'`{name}`' for name in missing)
                + '.')

        super().__init__(*args, **kwargs)

    @property
    def obsgeoloc(self):
        ''' Return the observer position relative to the geocenter '''
        earth_location, _ = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return self.obsbaryloc - earth_location

    @property
    def obsgeovel(self):
        ''' Return the observer velocity relative to the geocenter '''
        _, earth_velocity = get_body_barycentric_posvel(
            'earth',
            self.obstime,
        )
        return self.obsbaryvel - earth_velocity


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    ICRS,
    BCRSObserver,
)
def _icrs_to_bcrs_observer(icrs_coordinate, observer_frame):
    ''' Transform ICRS coordinates using the configured observer state '''
    coordinate = icrs_coordinate.transform_to(_as_gcrs(observer_frame))
    return observer_frame.realize_frame(coordinate.data)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    BCRSObserver,
    ICRS,
)
def _bcrs_observer_to_icrs(observer_coordinate, icrs_frame):
    ''' Transform observer-centered coordinates back to ICRS '''
    coordinate = _as_gcrs(
        observer_coordinate,
        observer_coordinate.data,
    )
    return coordinate.transform_to(icrs_frame)


@frame_transform_graph.transform(
    FunctionTransformWithFiniteDifference,
    BCRSObserver,
    BCRSObserver,
)
def _bcrs_observer_to_bcrs_observer(
        observer_coordinate, observer_frame):
    ''' Transform between arbitrary BCRS observer frames '''
    source = _as_gcrs(
        observer_coordinate,
        observer_coordinate.data,
    )
    coordinate = source.transform_to(_as_gcrs(observer_frame))
    return observer_frame.realize_frame(coordinate.data)
