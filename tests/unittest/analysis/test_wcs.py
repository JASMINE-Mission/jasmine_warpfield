#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.wcs import WCS
import jax.numpy as jnp
import numpy as np
from pytest import approx, mark, raises

from warpfield import Detector, Exposure, Optics, Pointing, Telescope
from warpfield.calibration import ScaleCalibration
from warpfield.distortion import IdentityDistortion
from warpfield.projection import (
    EquidistantProjection,
    GnomonicProjection,
)
from warpfield.wcs import generate_wcs


def generate_telescope(projection):
    optics = Optics(
        projection,
        IdentityDistortion(),
        plate_scale=[2.0, 3.0],
    )
    detectors = (
        Detector(20.0, [1.0, -2.0], [0.01, 0.02]),
        Detector(-15.0, [-1.5, 0.5], [0.02, 0.015]),
    )
    return Telescope(optics, detectors)


def generate_exposure():
    return Exposure(
        Pointing(
            ra=[10.0, 20.0],
            dec=[30.0, -20.0],
            position_angle=[15.0, -25.0],
        ),
        ScaleCalibration(np.log([1.1, 0.9])),
    )


@mark.parametrize(
    'projection',
    [GnomonicProjection(), EquidistantProjection()],
)
def test_generate_wcs_roundtrip(projection):
    telescope = generate_telescope(projection)
    exposure = generate_exposure()

    nested = generate_wcs(exposure, telescope)

    assert len(nested) == 2
    assert all(len(row) == 2 for row in nested)
    assert all(
        isinstance(wcs, WCS)
        for row in nested
        for wcs in row
    )

    ra = np.array([10.01, 9.98, 10.03])
    dec = np.array([30.02, 29.99, 30.04])
    for detector_index, wcs in enumerate(nested[0]):
        actual = np.asarray(
            wcs.all_world2pix(np.column_stack([ra, dec]), 0)
        )
        expected = telescope(
            jnp.full(3, exposure.pointing.ra[0]),
            jnp.full(3, exposure.pointing.dec[0]),
            jnp.full(3, exposure.pointing.position_angle[0]),
            jnp.asarray(ra),
            jnp.asarray(dec),
            jnp.full((3, 1), 1.1),
            jnp.full(3, detector_index, dtype=int),
        )
        assert actual == approx(expected, abs=1e-8)


def test_generate_wcs_projection_codes():
    exposure = generate_exposure()

    tan = generate_wcs(
        exposure,
        generate_telescope(GnomonicProjection()),
    )[0][0]
    arc = generate_wcs(
        exposure,
        generate_telescope(EquidistantProjection()),
    )[0][0]

    assert list(tan.wcs.ctype) == ['RA---TAN', 'DEC--TAN']
    assert list(arc.wcs.ctype) == ['RA---ARC', 'DEC--ARC']


def test_generate_wcs_validation():
    telescope = generate_telescope(GnomonicProjection())
    exposure = generate_exposure()

    with raises(TypeError, match='Exposure'):
        generate_wcs(object(), telescope)
    with raises(TypeError, match='Telescope'):
        generate_wcs(exposure, object())
