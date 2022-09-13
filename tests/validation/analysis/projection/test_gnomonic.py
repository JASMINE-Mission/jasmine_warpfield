#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, assume, settings

from .util import WCSProjection, longitude, latitude
from warpfield.analysis.projection.gnomonic import *


class Gnomonic(WCSProjection):
    def __init__(self, a0, d0, scale=1):
        super().__init__(a0, d0, scale=scale)
        self.projection = 'TAN'


@settings(deadline=500)
@given(longitude(), latitude(), longitude(), latitude())
def test_gnomonic_conversion(tel_ra, tel_dec, ra, dec):
    telescope = Gnomonic(tel_ra, tel_dec)
    assume(0.0001 < telescope.separation(ra, dec) < 60.0)

    X, Y = telescope.convert(ra, dec)
    x, y = gnomonic_conversion(tel_ra, tel_dec, ra, dec)

    assert x == approx(X)
    assert y == approx(Y)
