#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pytest import approx
from hypothesis import given
from hypothesis import assume
from hypothesis.strategies import floats
import numpy as np

from warpfield.analysis.projection.gnomonic import *

def degree():
    return floats(0.0, 360.0)

def longitude():
    return floats(0, 360.0)

def latitude():
    return floats(-90.0, 90.0)

@given(degree())
def test_gnomonic(theta):
    rad = degree_to_radian(theta)
    assert rad == approx(theta * np.pi / 180)

@given(longitude(), latitude())
def test_gnomonic_x_null(lon, lat):
    lon = degree_to_radian(lon)
    lat = degree_to_radian(lat)
    x = gnomonic_x(lon, lat, lon, lat)
    assert x == approx(0.0)

@given(longitude(), latitude())
def test_gnomonic_y_null(lon, lat):
    lon = degree_to_radian(lon)
    lat = degree_to_radian(lat)
    y = gnomonic_y(lon, lat, lon, lat)
    assert y == approx(0.0)

@given(longitude(), latitude())
def test_gnomonic_z_null(lon, lat):
    lon = degree_to_radian(lon)
    lat = degree_to_radian(lat)
    z = gnomonic_z(lon, lat, lon, lat)
    assert z == approx(1.0)

@given(longitude(), latitude())
def test_gnomonic_conversion_null(lon, lat):
    lon = degree_to_radian(lon)
    lat = degree_to_radian(lat)
    X, Y = gnomonic_conversion(lon, lat, lon, lat)
    assert X == approx(0.0, abs=3e-6)
    assert Y == approx(0.0)

@given(longitude(), longitude())
def test_gnomonic_conversion_lon(lon_t, lon):
    assume(np.abs(lon_t - lon) < 45.0)
    lon_t = degree_to_radian(lon_t)
    lon = degree_to_radian(lon)
    lat = lat_t = 0.0
    X, Y = gnomonic_conversion(lon_t, lat_t, lon, lat)
    assert Y == approx(0.0)

@given(longitude(), latitude(), latitude())
def test_gnomonic_conversion_lat(lon_t, lat_t, lat):
    assume(np.abs(lat_t - lat) < 45.0)
    lon_t = degree_to_radian(lon_t)
    lat_t = degree_to_radian(lat_t)
    lat = degree_to_radian(lat)
    lon = lon_t
    X, Y = gnomonic_conversion(lon_t, lat_t, lon, lat)
    assert X == approx(0.0, abs=1e-3)

@given(longitude(), latitude(), longitude(), latitude(), degree())
def test_gnomonic_rotate(tel_ra, tel_dec, ra, dec, pa):
    scale = 1.0e-6
    X0, Y0 = gnomonic(tel_ra, tel_dec, pa, ra, dec, scale)

    X1, Y1 = gnomonic(tel_ra, tel_dec, pa + 180.0, ra, dec, scale)
    assert X0 + X1 == approx(0.0, abs=1e-3)
    assert Y0 + Y1 == approx(0.0, abs=1e-3)
