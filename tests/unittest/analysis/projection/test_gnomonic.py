#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
import numpy as np

from warpfield.analysis.projection.gnomonic import *


def test_degree_to_radian():
    args = [-180, -90, 0, 90, 180]
    for theta in args:
        rad = degree_to_radian(theta)
        assert rad == approx(theta * np.pi / 180)


def test_gnomonic_conversion_zero():
    X, Y = gnomonic_conversion(0.0, 0.0, 0.0, 0.0)
    assert X == approx(0.0)
    assert Y == approx(0.0)


def test_gnomonic_conversion_lon():
    args = [-0.1, 0.0, 0.1]
    for lon in args:
        Y = gnomonic_conversion(0.0, 0.0, lon, 0.0)[1]
        assert Y == approx(0.0)


def test_gnomonic_conversion_lat():
    args = [-0.1, 0.0, 0.1]
    for lat in args:
        X = gnomonic_conversion(0.0, 0.0, 0.0, lat)[0]
        assert X == approx(0.0)


def test_gnomonic_rotate():
    def gnomonic_rotate(pa):
        a0 = 266.415  # Right Ascension of the Galactic Center
        d0 = -29.006  # Declination of the Galactic Center
        return gnomonic(a0, d0, pa, a0 + 1.0, d0 + 1.0, 1.0)

    X0, Y0 = gnomonic_rotate(0.0)

    X1, Y1 = gnomonic_rotate(90.0)
    assert X0 + Y1 == approx(0.0)
    assert Y0 - X1 == approx(0.0)

    X1, Y1 = gnomonic_rotate(180.0)
    assert X0 + X1 == approx(0.0)
    assert Y0 + Y1 == approx(0.0)
