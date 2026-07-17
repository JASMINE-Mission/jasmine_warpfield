#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.coordinates import BaseCoordinateFrame, BaseRADecFrame
from astropy.coordinates import GCRS, ICRS, SkyCoord
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
import astropy.units as u
import numpy as np
from pytest import approx, mark, raises

from warpfield.observer import (
    BaryCentric,
    BCRSObserver,
    GeoCentric,
    GeoCentricInertial,
    Observer,
    SSOObserver,
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


def test_geocentric_inertial_cancels_earth_velocity():
    epoch = Time('2025-01-01')
    observer = GeoCentricInertial(epoch)
    _, earth_velocity = get_body_barycentric_posvel('earth', epoch)

    assert isinstance(observer, Observer)
    assert isinstance(observer, BaseRADecFrame)
    assert not isinstance(observer, GCRS)
    assert observer.obstime == epoch
    assert observer.obsgeoloc.xyz.to_value(u.m) == approx([0.0, 0.0, 0.0])
    assert observer.obsgeovel.xyz.to_value(u.m / u.s) == approx(
        -earth_velocity.xyz.to_value(u.m / u.s))


def test_geocentric_inertial_tracks_obstime():
    observer = GeoCentricInertial(Time('2025-01-01'))
    replicated = observer.replicate_without_data(
        obstime=Time('2025-07-01'),
    )
    _, earth_velocity = get_body_barycentric_posvel(
        'earth',
        replicated.obstime,
    )

    assert replicated.obsgeovel.xyz.to_value(u.m / u.s) == approx(
        -earth_velocity.xyz.to_value(u.m / u.s))
    assert replicated.obsgeovel.xyz.to_value(u.m / u.s) != approx(
        observer.obsgeovel.xyz.to_value(u.m / u.s))


def test_geocentric_inertial_matches_configured_gcrs():
    epoch = Time('2025-01-01')
    _, earth_velocity = get_body_barycentric_posvel('earth', epoch)
    coordinate = SkyCoord(
        ra=[10.0, 20.0] * u.deg,
        dec=[-5.0, 15.0] * u.deg,
        distance=[10.0, 20.0] * u.pc,
        frame=ICRS(),
    )

    expected = coordinate.transform_to(GCRS(
        obstime=epoch,
        obsgeovel=-earth_velocity,
    ))
    actual = coordinate.transform_to(GeoCentricInertial(epoch))

    assert actual.ra.degree == approx(expected.ra.degree)
    assert actual.dec.degree == approx(expected.dec.degree)
    assert actual.distance.to_value(u.pc) == approx(
        expected.distance.to_value(u.pc))


def test_sso_observer_default_orbit():
    observer = SSOObserver(Time('2025-01-01'))
    location = observer.obsgeoloc.xyz.to_value(u.m)
    velocity = observer.obsgeovel.xyz.to_value(u.m / u.s)
    delta_raan = (
        observer.raan - observer.mean_sun_right_ascension
    ).wrap_at(180 * u.deg)
    local_time = (
        12 + delta_raan.to_value(u.deg) / 15
    ) % 24

    assert observer.altitude.to_value(u.km) == approx(600.0)
    assert observer.phase.to_value(u.one) == approx(0.0)
    assert observer.ltan.to_value(u.hourangle) == approx(6.0)
    assert observer.inclination.to_value(u.deg) == approx(97.78761565)
    assert observer.orbital_period.to_value(u.min) == approx(96.68643251)
    assert np.linalg.norm(location) == approx(
        observer.orbital_radius.to_value(u.m))
    assert np.linalg.norm(velocity) == approx(
        observer.orbital_velocity.to_value(u.m / u.s))
    assert location @ velocity == approx(0.0, abs=1.0e-5)
    assert location[2] == approx(0.0)
    assert velocity[2] > 0
    assert local_time == approx(6.0)


def test_sso_observer_ltan_rotates_ascending_node():
    epoch = Time('2025-01-01')
    morning = SSOObserver(epoch, ltan=6 * u.hourangle)
    evening = SSOObserver(epoch, ltan=18 * u.hourangle)
    separation = (
        evening.raan - morning.raan
    ).wrap_at(360 * u.deg)

    assert separation.to_value(u.deg) == approx(180.0)
    assert evening.obsgeoloc.xyz.to_value(u.m) == approx(
        -morning.obsgeoloc.xyz.to_value(u.m))


def test_sso_observer_barycentric_state():
    observer = SSOObserver(Time('2025-01-01'), phase=0.25)
    earth_location, earth_velocity = get_body_barycentric_posvel(
        'earth',
        observer.obstime,
    )

    assert observer.obsbaryloc.xyz.to_value(u.au) == approx(
        (earth_location + observer.obsgeoloc).xyz.to_value(u.au))
    assert observer.obsbaryvel.xyz.to_value(u.km / u.s) == approx(
        (earth_velocity + observer.obsgeovel).xyz.to_value(u.km / u.s))


def test_sso_observer_matches_configured_gcrs():
    epoch = Time('2025-01-01')
    observer = SSOObserver(
        epoch,
        phase=0.25,
        altitude=700 * u.km,
        ltan=10.5 * u.hourangle,
    )
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


def test_sso_observer_validation():
    epoch = Time('2025-01-01')

    with raises(ValueError, match='altitude'):
        SSOObserver(epoch, altitude=0 * u.km)
    with raises(ValueError, match='phase'):
        SSOObserver(epoch, phase=np.nan)
    with raises(ValueError, match='ltan'):
        SSOObserver(epoch, ltan=24 * u.hourangle)


@mark.parametrize('frame', [
    BaryCentric,
    GeoCentric,
    GeoCentricInertial,
    SSOObserver,
])
def test_observer_requires_obstime(frame):
    with raises(TypeError, match='obstime.*required'):
        frame()


@mark.parametrize('frame', [
    GeoCentric,
    GeoCentricInertial,
])
@mark.parametrize('attribute', [
    'obsgeoloc',
    'obsgeovel',
])
def test_geocentric_state_is_read_only(frame, attribute):
    with raises(TypeError, match='unexpected keywords'):
        frame(Time('2025-01-01'), **{attribute: [0.0, 0.0, 0.0]})
