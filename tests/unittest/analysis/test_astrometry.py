#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield import (
    Astrometry,
    Detector,
    Exposure,
    Measurement,
    Optics,
    Pointing,
    SourceCatalog,
    Telescope,
)
from warpfield.calibration import ScaleCalibration
from warpfield.distortion import IdentityDistortion
from warpfield.projection import GnomonicProjection


def generate_astrometry():
    source = SourceCatalog(
        ra=[266.5, 266.4],
        dec=[-28.9, -29.2],
    )
    pointing = Pointing(
        ra=[266.4, 266.5],
        dec=[-29.0, -29.1],
        position_angle=[0.0, 5.0],
    )
    calibration = ScaleCalibration(jnp.log(jnp.array([1.0, 1.1])))
    exposure = Exposure(pointing, calibration)
    optics = Optics(
        GnomonicProjection(), IdentityDistortion(), plate_scale=[2.0, 3.0])
    detectors = (
        Detector(0.0, [0.0, 0.0], [1.0, 1.0]),
        Detector(90.0, [1.0, 2.0], [0.5, 2.0]),
    )
    return Astrometry(
        source,
        Telescope(optics, detectors),
        exposure,
    )


def generate_measurement(astrometry):
    source_index = jnp.array([0, 1, 0])
    exposure_index = jnp.array([0, 1, 1])
    detector_index = jnp.array([0, 1, 0])
    ra, dec = astrometry.source.take(source_index)
    tel_ra, tel_dec, tel_pa, scale_factor = astrometry.exposure.take(
        exposure_index)
    expected = astrometry.telescope(
        tel_ra,
        tel_dec,
        tel_pa,
        ra,
        dec,
        scale_factor,
        detector_index,
    )
    offset = jnp.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    measurement = Measurement(
        expected + offset,
        source_index,
        exposure_index,
        detector_index,
    )
    return measurement, expected, offset


def test_astrometry():
    astrometry = generate_astrometry()
    measurement, expected, offset = generate_measurement(astrometry)

    assert isinstance(astrometry, zdx.Base)
    assert astrometry(measurement) == approx(expected)
    assert astrometry.residual(measurement) == approx(offset)
    assert eqx.filter_jit(astrometry)(measurement) == approx(expected)
    assert eqx.filter_jit(astrometry.residual)(measurement) == approx(offset)


def test_astrometry_gradient_excludes_measurement():
    astrometry = generate_astrometry()
    measurement, _, _ = generate_measurement(astrometry)

    def loss(parameters, data):
        return jnp.sum(parameters.residual(data)**2)

    gradient = eqx.filter_grad(loss)(astrometry, measurement)

    assert jnp.isfinite(gradient.source.ra).all()
    assert jnp.isfinite(gradient.source.dec).all()
    assert jnp.isfinite(gradient.exposure.pointing.ra).all()
    assert jnp.isfinite(gradient.exposure.calibration.coefficient).all()
    assert jnp.isfinite(gradient.telescope.optics.plate_scale).all()
    assert jnp.isfinite(gradient.telescope.detectors[0].offset).all()
    assert jnp.isfinite(gradient.telescope.detectors[1].offset).all()


def test_astrometry_zodiax_update():
    astrometry = generate_astrometry()
    updated = astrometry.set('source.ra', jnp.array([1.0, 2.0]))

    assert astrometry.get('source.ra') == approx([266.5, 266.4])
    assert updated.get('source.ra') == approx([1.0, 2.0])

    path = 'exposure.calibration.coefficient'
    updated = astrometry.set(path, jnp.array([0.1, 0.2]))

    assert astrometry.get(path) == approx(jnp.log(jnp.array([1.0, 1.1])))
    assert updated.get(path) == approx([0.1, 0.2])


def test_astrometry_validation():
    astrometry = generate_astrometry()

    with raises(TypeError, match='SourceCatalog'):
        Astrometry(object(), astrometry.telescope, astrometry.exposure)
    with raises(TypeError, match='Telescope'):
        Astrometry(astrometry.source, object(), astrometry.exposure)
    with raises(TypeError, match='Exposure'):
        Astrometry(astrometry.source, astrometry.telescope, object())
    with raises(TypeError, match='Measurement'):
        astrometry(object())
