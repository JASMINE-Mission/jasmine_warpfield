#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given, assume, settings
from hypothesis.strategies import floats
import numpy as np

from warpfield.analysis.projection.gnomonic import *


def degree():
    return floats(0.0, 360.0)


def longitude(unit='rad'):
    if unit == 'rad':
        return floats(0, 2 * np.pi)
    else:
        return floats(0, 360.0)


def latitude(unit='rad'):
    if unit == 'rad':
        return floats(-np.pi / 2.0, np.pi / 2.0)
    else:
        return floats(-90.0, 90.0)


def get_xyz(a, d, unit='rad'):
    if unit != 'rad':
        a = degree_to_radian(a)
        d = degree_to_radian(d)
    return np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)


def dot(a0, d0, a1, d1, unit='rad'):
    x0, y0, z0 = get_xyz(a0, d0, unit)
    x1, y1, z1 = get_xyz(a1, d1, unit)
    return x0 * x1 + y0 * y1 + z0 * z1


def cosd(theta):
    return np.cos(theta * np.pi / 180.0)


@given(floats(0.7, 1.0))
def test_rsinr(rho):
    assume(rho < 1.0)
    assert rsinr(rho) == approx(np.arccos(rho) / np.sqrt(1.0 - rho**2))


@settings(max_examples=10)
@given(degree())
def test_degree_to_radian(theta):
    rad = degree_to_radian(theta)
    assert rad == approx(theta * np.pi / 180)


@settings(max_examples=10)
@given(longitude(), latitude())
def test_gnomonic_conversion_null(lon, lat):
    X, Y = gnomonic_conversion(lon, lat, lon, lat)
    assert X == approx(0.0)
    assert Y == approx(0.0)


@settings(max_examples=10)
@given(longitude(), longitude())
def test_gnomonic_conversion_lon(lon_t, lon):
    lat = lat_t = 0.0
    assume(dot(lon_t, lat_t, lon, lat) > cosd(30.0))
    Y = gnomonic_conversion(lon_t, lat_t, lon, lat)[1]
    assert Y == approx(0.0)


@settings(max_examples=10)
@given(longitude(), latitude(), latitude())
def test_gnomonic_conversion_lat(lon_t, lat_t, lat):
    lon = lon_t
    assume(dot(lon_t, lat_t, lon, lat) > cosd(30.0))
    X = gnomonic_conversion(lon_t, lat_t, lon, lat)[0]
    assert X == approx(0.0)


@settings(deadline=500)
@given(longitude('deg'), latitude('deg'), longitude('deg'), latitude('deg'))
def test_gnomonic_rotate(tel_ra, tel_dec, ra, dec):
    pa = 0.0
    scale = 1.0e-6
    assume(dot(tel_ra, tel_dec, ra, dec, 'deg') > cosd(30.0))
    X0, Y0 = gnomonic(tel_ra, tel_dec, pa, ra, dec, scale)

    X1, Y1 = gnomonic(tel_ra, tel_dec, pa + 90.0, ra, dec, scale)
    assert X0 - Y1 == approx(0.0, abs=1e-6)
    assert Y0 + X1 == approx(0.0, abs=1e-6)

    X1, Y1 = gnomonic(tel_ra, tel_dec, pa + 180.0, ra, dec, scale)
    assert X0 + X1 == approx(0.0, abs=1e-6)
    assert Y0 + Y1 == approx(0.0, abs=1e-6)
