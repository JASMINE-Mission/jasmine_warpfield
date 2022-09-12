#!/usr/bin/env python
# -*- coding: utf-8 -*-
from astropy.coordinates import SkyCoord
from hypothesis.strategies import floats
from hypothesis import HealthCheck
import numpy as np

from warpfield.telescope.util import get_projection


suppress_too_much_filter = [HealthCheck.filter_too_much]


def longitude():
    return floats(0, 2 * np.pi, exclude_max=True)


def latitude():
    return floats(
        -np.pi / 2, np.pi / 2, exclude_min=True, exclude_max=True)


class WCSProjection:
    def __init__(self, a0, d0, scale=1):
        self.tel_ra  = self.radian_to_degree(a0)
        self.tel_dec = self.radian_to_degree(d0)
        self.tel = SkyCoord(self.tel_ra, self.tel_dec, unit='deg')
        self.scale = scale
        self.projection = 'TAN'

    @property
    def proj(self):
        return get_projection(self.tel, self.scale, name=self.projection)

    @staticmethod
    def radian_to_degree(rad):
        return rad / np.pi * 180.0

    @staticmethod
    def object(ra, dec):
        return SkyCoord(ra, dec, unit='rad')

    def separation(self, ra, dec):
        return self.tel.separation(self.object(ra, dec)).deg

    def convert(self, ra, dec):
        obj = self.object(ra, dec)
        return obj.to_pixel(self.proj, origin=0)
