#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.coordinates import BaseCoordinateFrame, GCRS, ICRS, SkyCoord
from astropy.time import Time
import astropy.units as u
from pytest import approx

from warpfield.analysis.observer import GeoCentric, Observer


def test_geocentric_is_observer_frame():
    epoch = Time('2025-01-01')
    observer = GeoCentric(epoch)

    assert isinstance(observer, Observer)
    assert isinstance(observer, BaseCoordinateFrame)
    assert isinstance(observer, GCRS)
    assert observer.obstime == epoch
    assert observer.obsgeoloc.xyz.to_value(u.m) == approx([0.0, 0.0, 0.0])
    assert observer.obsgeovel.xyz.to_value(u.m / u.s) == approx(
        [0.0, 0.0, 0.0])


def test_geocentric_matches_astropy_gcrs():
    epoch = Time('2025-01-01')
    position = [100.0, 200.0, 300.0] * u.km
    velocity = [1.0, 2.0, 3.0] * u.km / u.s
    coordinate = SkyCoord(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        distance=[10.0, 20.0] * u.pc,
        frame=ICRS(),
    )

    expected = coordinate.transform_to(GCRS(
        obstime=epoch,
        obsgeoloc=position,
        obsgeovel=velocity,
    ))
    actual = coordinate.transform_to(GeoCentric(
        epoch,
        obsgeoloc=position,
        obsgeovel=velocity,
    ))

    assert actual.ra.degree == approx(expected.ra.degree)
    assert actual.dec.degree == approx(expected.dec.degree)
    assert actual.distance.to_value(u.pc) == approx(
        expected.distance.to_value(u.pc))


def test_geocentric_roundtrip():
    coordinate = SkyCoord(
        ra=[10.0] * u.deg,
        dec=[-5.0] * u.deg,
        distance=[10.0] * u.pc,
        frame=ICRS(),
    )

    roundtrip = coordinate.transform_to(
        GeoCentric(Time('2025-01-01'))).transform_to(ICRS())

    assert roundtrip.ra.degree == approx(coordinate.ra.degree, abs=1e-10)
    assert roundtrip.dec.degree == approx(coordinate.dec.degree, abs=1e-10)
