#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
from pytest import approx, mark, raises

from warpfield import Detector, Exposure, Optics, Pointing, Telescope
from warpfield.calibration import (
    IdentityCalibration,
    ScaleCalibration,
)
from warpfield.distortion import IdentityDistortion
from warpfield.footprint import (
    detector_footprint,
    telescope_footprints,
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
    optics = Optics(
        projection,
        IdentityDistortion(),
        plate_scale=[1.0, 1.2],
    )
    detector = Detector(
        rotation=20.0,
        offset=[0.01, -0.02],
        pixel_scale=[0.002, 0.003],
        shape=(20, 16),
    )
    return Telescope(optics, (detector,))


def test_detector_footprint():
    footprint = detector_footprint(generate_detector())

    assert footprint == approx(np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 4.0],
        [0.0, 4.0],
        [0.0, 0.0],
    ]))


def test_detector_footprint_sampling():
    footprint = detector_footprint(
        generate_detector(),
        samples_per_edge=3,
    )

    assert footprint.shape == (13, 2)
    assert footprint[0] == approx(footprint[-1])


@mark.parametrize(
    'projection',
    [GnomonicProjection(), EquidistantProjection()],
)
def test_telescope_footprints_roundtrip(projection):
    telescope = generate_telescope(projection)
    exposures = Exposure(
        Pointing([10.0, 20.0], [30.0, 40.0], [15.0, 25.0]),
        ScaleCalibration([np.log(1.1), 0.0]),
    )
    exposure = exposures[0]

    sky = telescope_footprints(
        telescope,
        exposure,
        samples_per_edge=2,
        limit=False,
    )[0]
    size = len(sky)
    focal_plane = telescope.focal_plane(
        jnp.full(size, exposure.pointing.ra[0]),
        jnp.full(size, exposure.pointing.dec[0]),
        jnp.full(size, exposure.pointing.position_angle[0]),
        jnp.asarray(sky.icrs.ra.degree),
        jnp.asarray(sky.icrs.dec.degree),
        jnp.full((size, 1), 1.1),
    )
    expected = detector_footprint(
        telescope.detectors[0],
        samples_per_edge=2,
    )

    assert focal_plane == approx(expected, abs=1e-7)


def test_telescope_footprints_galactic():
    telescope = generate_telescope(GnomonicProjection())
    exposure = Exposure(
        Pointing([10.0], [30.0], [15.0]),
        IdentityCalibration(),
    )

    footprint = telescope_footprints(
        telescope,
        exposure,
        frame='galactic',
    )[0]

    assert footprint.frame.name == 'galactic'


def test_footprint_validation():
    with raises(TypeError, match='Detector'):
        detector_footprint(object())
    with raises(ValueError, match='positive'):
        detector_footprint(generate_detector(), samples_per_edge=0)

    telescope = generate_telescope(GnomonicProjection())
    exposure = Exposure(
        Pointing([10.0, 20.0], [30.0, 40.0], [15.0, 25.0]),
        IdentityCalibration(),
    )
    with raises(ValueError, match='exactly one'):
        telescope_footprints(telescope, exposure)
    with raises(ValueError, match='frame'):
        telescope_footprints(telescope, exposure[0], frame='ecliptic')
