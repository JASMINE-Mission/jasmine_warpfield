#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import (
    Astrometry,
    Detector,
    Observation,
    Optics,
    Pointing,
    SourceCatalog,
    Telescope,
)
from warpfield.analysis.distortion import IdentityDistortion
from warpfield.analysis.projection import GnomonicProjection


def generate_astrometry():
    source = SourceCatalog(
        ra=[266.5, 266.4],
        dec=[-28.9, -29.2],
    )
    pointing = Pointing(
        ra=[266.4, 266.5],
        dec=[-29.0, -29.1],
        position_angle=[0.0, 5.0],
        scale=[[1.0, 1.0], [2.0, 3.0]],
    )
    optics = Optics(GnomonicProjection(), IdentityDistortion())
    detector = Detector(
        rotation=[0.0, 90.0],
        offset=[[0.0, 0.0], [1.0, 2.0]],
        pixel_scale=[[1.0, 1.0], [0.5, 2.0]],
    )
    return Astrometry(
        source,
        Telescope(pointing, optics, detector),
    )


def generate_observation(astrometry):
    source_index = jnp.array([0, 1, 0])
    pointing_index = jnp.array([0, 1, 1])
    detector_index = jnp.array([0, 1, 0])
    ra, dec = astrometry.source.take(source_index)
    expected = astrometry.telescope(
        ra, dec, pointing_index, detector_index)
    offset = jnp.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
    observation = Observation(
        expected + offset,
        source_index,
        pointing_index,
        detector_index,
    )
    return observation, expected, offset


def test_astrometry():
    astrometry = generate_astrometry()
    observation, expected, offset = generate_observation(astrometry)

    assert isinstance(astrometry, zdx.Base)
    assert astrometry(observation) == approx(expected)
    assert astrometry.residual(observation) == approx(offset)
    assert eqx.filter_jit(astrometry)(observation) == approx(expected)
    assert eqx.filter_jit(astrometry.residual)(observation) == approx(offset)


def test_astrometry_gradient():
    astrometry = generate_astrometry()
    observation, _, _ = generate_observation(astrometry)

    def loss(value):
        return jnp.sum(value.residual(observation)**2)

    gradient = eqx.filter_grad(loss)(astrometry)

    assert jnp.isfinite(gradient.source.ra).all()
    assert jnp.isfinite(gradient.source.dec).all()
    assert jnp.isfinite(gradient.telescope.pointing.ra).all()
    assert jnp.isfinite(gradient.telescope.detector.offset).all()


def test_astrometry_zodiax_update():
    astrometry = generate_astrometry()
    updated = astrometry.set('source.ra', jnp.array([1.0, 2.0]))

    assert astrometry.get('source.ra') == approx([266.5, 266.4])
    assert updated.get('source.ra') == approx([1.0, 2.0])


def test_astrometry_validation():
    astrometry = generate_astrometry()

    with raises(TypeError, match='SourceCatalog'):
        Astrometry(object(), astrometry.telescope)
    with raises(TypeError, match='Telescope'):
        Astrometry(astrometry.source, object())
    with raises(TypeError, match='Observation'):
        astrometry(object())
