#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Base class for observer-centered celestial coordinate frames '''

from astropy.coordinates import BaseCoordinateFrame, TimeAttribute


__all__ = ['Observer']


class Observer(BaseCoordinateFrame):
    ''' Base class for observer-centered coordinate frames '''

    obstime = TimeAttribute(default=None)

    def __init__(self, *args, **kwargs):
        if kwargs.get('obstime') is None:
            raise TypeError('`obstime` is required for an Observer frame.')
        super().__init__(*args, **kwargs)
