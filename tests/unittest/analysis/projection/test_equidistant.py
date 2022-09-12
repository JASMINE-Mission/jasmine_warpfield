#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx

from warpfield.analysis.projection.equidistant import *


def test_equidistant_conversion_zero():
    X, Y = equidistant_conversion(0.0, 0.0, 0.0, 0.0)
    assert X == approx(0.0)
    assert Y == approx(0.0)


def test_equidistant_conversion_lon():
    args = [-0.1, 0.0, 0.1]
    for lon in args:
        Y = equidistant_conversion(0.0, 0.0, lon, 0.0)[1]
        assert Y == approx(0.0)


def test_equidistant_conversion_lat():
    args = [-0.1, 0.0, 0.1]
    for lat in args:
        X = equidistant_conversion(0.0, 0.0, 0.0, lat)[0]
        assert X == approx(0.0)


def test_equidistant_rotate():
    def equidistant_rotate(pa):
        a0 = 266.415  # Right Ascension of the Galactic Center
        d0 = -29.006  # Declination of the Galactic Center
        return equidistant(a0, d0, pa, a0 + 1.0, d0 + 1.0, 1.0)

    X0, Y0 = equidistant_rotate(0.0)

    X1, Y1 = equidistant_rotate(90.0)
    assert X0 + Y1 == approx(0.0)
    assert Y0 - X1 == approx(0.0)

    X1, Y1 = equidistant_rotate(180.0)
    assert X0 + X1 == approx(0.0)
    assert Y0 + Y1 == approx(0.0)
