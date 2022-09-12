#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, assume, settings
from hypothesis import HealthCheck as hc
from hypothesis.strategies import floats
from astropy.coordinates import SkyCoord
import numpy as np

from warpfield.telescope.util import get_projection
from warpfield.analysis.projection.gnomonic import *


def degree():
    return floats(0.0, 360.0)


def longitude():
    return floats(0, 2 * np.pi, exclude_max=True)


def latitude():
    return floats(
        -np.pi / 2.0, np.pi / 2.0, exclude_min=True, exclude_max=True)


def radian_to_degree(rad):
    return rad / np.pi * 180.0


class Gnomonic:
    def __init__(self, a0, d0, scale=1):
        self.tel_ra  = radian_to_degree(a0)
        self.tel_dec = radian_to_degree(d0)
        self.tel = SkyCoord(self.tel_ra, self.tel_dec, unit='deg')
        self.proj = get_projection(self.tel, scale)

    @staticmethod
    def object(ra, dec):
        return SkyCoord(ra, dec, unit='rad')

    def separation(self, ra, dec):
        return self.tel.separation(self.object(ra, dec)).deg

    def convert(self, ra, dec):
        obj = self.object(ra, dec)
        return obj.to_pixel(self.proj, origin=0)


@given(degree())
def test_degree_to_radian(theta):
    rad = degree_to_radian(theta)
    assert rad == approx(theta * np.pi / 180)


@settings(deadline=500, suppress_health_check=[hc.filter_too_much])
@given(longitude(), latitude(), longitude(), latitude())
def test_gnomonic_conversion(tel_ra, tel_dec, ra, dec):
    telescope = Gnomonic(tel_ra, tel_dec)
    assume(np.abs(tel_dec) < np.pi / 2 \
        and 0.0001 < telescope.separation(ra, dec) < 60.0)

    X, Y = telescope.convert(ra, dec)
    x, y = gnomonic_conversion(tel_ra, tel_dec, ra, dec)

    assert x == approx(X)
    assert y == approx(Y)
