#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.coordinates import BaseCoordinateFrame, BaseRADecFrame
from astropy.coordinates import GCRS, ICRS, SkyCoord
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
from pytest import approx, raises

from warpfield.observer import (
    BaryCentric,
    BCRSObserver,
    GeoCentric,
    Observer,
)


def test_bcrs_observer_requires_state():
    epoch = Time('2025-01-01')
    location = [1.0, 2.0, 3.0] * u.au
    velocity = [10.0, 20.0, 30.0] * u.km / u.s

    with raises(TypeError, match='obsbaryloc.*obsbaryvel'):
        BCRSObserver(epoch)
    with raises(TypeError, match='obsbaryvel'):
        BCRSObserver(epoch, location)
    with raises(TypeError, match='obstime'):
        BCRSObserver(
            obsbaryloc=location,
            obsbaryvel=velocity,
        )


def test_bcrs_observer_geocentric_state():
    epoch = Time('2025-01-01')
    location = [1.0, 2.0, 3.0] * u.au
    velocity = [10.0, 20.0, 30.0] * u.km / u.s
    observer = BCRSObserver(epoch, location, velocity)
    earth_location, earth_velocity = get_body_barycentric_posvel(
        'earth',
        epoch,
    )

    assert isinstance(observer, Observer)
    assert observer.obsbaryloc.xyz.to_value(u.au) == approx(location.value)
    assert observer.obsbaryvel.xyz.to_value(u.km / u.s) == approx(
        velocity.value)
    assert observer.obsgeoloc.xyz.to_value(u.au) == approx(
        (observer.obsbaryloc - earth_location).xyz.to_value(u.au))
    assert observer.obsgeovel.xyz.to_value(u.km / u.s) == approx(
        (observer.obsbaryvel - earth_velocity).xyz.to_value(u.km / u.s))


def test_bcrs_observer_matches_configured_gcrs():
    epoch = Time('2025-01-01')
    location = [1.0, 2.0, 3.0] * u.au
    velocity = [10.0, 20.0, 30.0] * u.km / u.s
    observer = BCRSObserver(epoch, location, velocity)
    coordinate = SkyCoord(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        distance=[10.0, 20.0] * u.pc,
        frame=ICRS(),
    )

    expected = coordinate.transform_to(GCRS(
        obstime=epoch,
        obsgeoloc=observer.obsgeoloc,
        obsgeovel=observer.obsgeovel,
    ))
    actual = coordinate.transform_to(observer)

    assert actual.ra.degree == approx(expected.ra.degree)
    assert actual.dec.degree == approx(expected.dec.degree)
    assert actual.distance.to_value(u.pc) == approx(
        expected.distance.to_value(u.pc))


def test_bcrs_observer_tracks_obstime():
    observer = BCRSObserver(
        Time('2025-01-01'),
        [1.0, 2.0, 3.0] * u.au,
        [10.0, 20.0, 30.0] * u.km / u.s,
    )
    replicated = observer.replicate_without_data(
        obstime=Time('2025-07-01'),
    )

    assert replicated.obsbaryloc == observer.obsbaryloc
    assert replicated.obsbaryvel == observer.obsbaryvel
    assert replicated.obsgeoloc != observer.obsgeoloc
    assert replicated.obsgeovel != observer.obsgeovel


def test_barycentric_is_observer_frame():
    epoch = Time('2025-01-01')
    observer = BaryCentric(epoch)

    assert isinstance(observer, Observer)
    assert isinstance(observer, BaseCoordinateFrame)
    assert isinstance(observer, ICRS)
    assert observer.obstime == epoch


def test_barycentric_matches_icrs():
    coordinate = SkyCoord(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        distance=[10.0, 20.0] * u.pc,
        frame=ICRS(),
    )

    actual = coordinate.transform_to(BaryCentric(Time('2025-01-01')))
    roundtrip = actual.transform_to(ICRS())

    assert actual.ra.degree == approx(coordinate.ra.degree)
    assert actual.dec.degree == approx(coordinate.dec.degree)
    assert actual.distance.to_value(u.pc) == approx(
        coordinate.distance.to_value(u.pc))
    assert roundtrip.ra.degree == approx(coordinate.ra.degree)
    assert roundtrip.dec.degree == approx(coordinate.dec.degree)


def test_geocentric_is_observer_frame():
    epoch = Time('2025-01-01')
    observer = GeoCentric(epoch)

    assert isinstance(observer, Observer)
    assert isinstance(observer, BaseCoordinateFrame)
    assert isinstance(observer, BaseRADecFrame)
    assert not isinstance(observer, GCRS)
    assert observer.obstime == epoch
    assert observer.obsgeoloc.xyz.to_value(u.m) == approx([0.0, 0.0, 0.0])
    assert observer.obsgeovel.xyz.to_value(u.m / u.s) == approx(
        [0.0, 0.0, 0.0])


def test_geocentric_matches_astropy_gcrs():
    epoch = Time('2025-01-01')
    coordinate = SkyCoord(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        distance=[10.0, 20.0] * u.pc,
        frame=ICRS(),
    )

    expected = coordinate.transform_to(GCRS(obstime=epoch))
    actual = coordinate.transform_to(GeoCentric(epoch))

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
