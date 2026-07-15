#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Base class for observer-centered celestial coordinate frames '''

from astropy.coordinates import BaseCoordinateFrame, TimeAttribute
from astropy.coordinates.builtin_frames import utils


__all__ = ['Observer']


class Observer(BaseCoordinateFrame):
    ''' Base class for observer-centered coordinate frames '''

    obstime = TimeAttribute(default=utils.DEFAULT_OBSTIME)
