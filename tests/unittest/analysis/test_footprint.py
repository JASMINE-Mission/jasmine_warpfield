#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
from pytest import approx, mark, raises

from warpfield import Detector, Exposure, Pointing, Telescope
from warpfield.footprint import (
    celestial_footprints,
    focalplane_footprints,
)
from warpfield.projection import (
    EquidistantProjection,
    GnomonicProjection,
)


def generate_detector():
    return Detector(
        rotation=0.0,
        offset=[1.0, 2.0],
        pixel_scale=[0.5, 2.0],
        shape=(4, 2),
    )


def generate_telescope(projection):
    detector = Detector(
        rotation=20.0,
        offset=[0.01, -0.02],
        pixel_scale=[0.002, 0.003],
        shape=(20, 16),
    )
    return Telescope(projection, [1.0, 1.2], (detector,))


def test_focalplane_footprints_from_detector():
    footprints = focalplane_footprints(generate_detector())

    assert isinstance(footprints, tuple)
    assert len(footprints) == 1
    assert footprints[0] == approx(np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 4.0],
        [0.0, 4.0],
        [0.0, 0.0],
    ]))


def test_focalplane_footprints_sampling():
    footprint = focalplane_footprints(
        generate_detector(),
        samples_per_edge=3,
    )[0]

    assert footprint.shape == (13, 2)
    assert footprint[0] == approx(footprint[-1])


def test_focalplane_footprints_from_tuple_and_telescope():
    telescope = generate_telescope(GnomonicProjection())
    detector = generate_detector()

    from_tuple = focalplane_footprints((detector, *telescope.detectors))
    from_telescope = focalplane_footprints(telescope)

    assert len(from_tuple) == 2
    assert len(from_telescope) == 1
    assert from_tuple[1] == approx(from_telescope[0])


@mark.parametrize(
    'projection',
    [GnomonicProjection(), EquidistantProjection()],
)
def test_celestial_footprints_roundtrip(projection):
    telescope = generate_telescope(projection)
    pointings = Pointing(
        [10.0, 20.0],
        [30.0, 40.0],
        [15.0, 25.0],
    )
    pointing = pointings[0]

    sky = celestial_footprints(
        telescope,
        pointing,
        samples_per_edge=2,
        limit=False,
    )[0]
    size = len(sky)
    focal_plane = telescope.focal_plane(
        jnp.full(size, pointing.ra[0]),
        jnp.full(size, pointing.dec[0]),
        jnp.full(size, pointing.position_angle[0]),
        jnp.asarray(sky.icrs.ra.degree),
        jnp.asarray(sky.icrs.dec.degree),
        jnp.ones((size, 1)),
    )
    expected = focalplane_footprints(
        telescope.detectors[0],
        samples_per_edge=2,
    )[0]

    assert focal_plane == approx(expected, abs=1e-7)


def test_celestial_footprints_galactic():
    telescope = generate_telescope(GnomonicProjection())
    pointing = Pointing([10.0], [30.0], [15.0])

    footprint = celestial_footprints(
        telescope,
        pointing,
        frame='galactic',
    )[0]

    assert footprint.frame.name == 'galactic'


def test_celestial_footprints_from_exposure():
    telescope = generate_telescope(GnomonicProjection())
    exposure = Exposure(Pointing([10.0], [30.0], [15.0]))

    from_pointing = celestial_footprints(
        telescope,
        exposure.pointing,
    )[0]
    from_exposure = celestial_footprints(
        telescope,
        exposure,
    )[0]

    assert from_exposure.ra.degree == approx(from_pointing.ra.degree)
    assert from_exposure.dec.degree == approx(from_pointing.dec.degree)


def test_footprint_validation():
    with raises(TypeError, match='Telescope, Detector'):
        focalplane_footprints(object())
    with raises(TypeError, match='Telescope, Detector'):
        focalplane_footprints((generate_detector(), object()))
    with raises(ValueError, match='positive'):
        focalplane_footprints(generate_detector(), samples_per_edge=0)

    telescope = generate_telescope(GnomonicProjection())
    pointing = Pointing(
        [10.0, 20.0],
        [30.0, 40.0],
        [15.0, 25.0],
    )
    with raises(ValueError, match='exactly one'):
        celestial_footprints(telescope, pointing)
    with raises(ValueError, match='frame'):
        celestial_footprints(telescope, pointing[0], frame='ecliptic')
    with raises(TypeError, match='Pointing or Exposure'):
        celestial_footprints(telescope, object())
