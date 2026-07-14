#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax.numpy as jnp
from pytest import approx, raises

from warpfield.analysis import Detector, Optics, Pointing, Telescope
from warpfield.analysis.distortion import LegendreDistortion
from warpfield.analysis.projection import GnomonicProjection


def generate_telescope():
    pointing = Pointing(
        ra=[266.4, 266.5],
        dec=[-29.0, -29.1],
        position_angle=[0.0, 5.0],
        scale=[[1.0, 1.0], [2.0, 3.0]],
    )
    distortion = LegendreDistortion(
        coeff_x=jnp.linspace(0.0, 0.1, 18),
        coeff_y=jnp.linspace(0.1, 0.0, 18),
        plane_scale=10.0,
    )
    optics = Optics(GnomonicProjection(), distortion)
    detector = Detector(
        rotation=[0.0, 90.0],
        offset=[[0.0, 0.0], [1.0, 2.0]],
        pixel_scale=[[1.0, 1.0], [0.5, 2.0]],
    )
    return Telescope(pointing, optics, detector)


def test_telescope_pipeline():
    telescope = generate_telescope()
    ra = jnp.array([266.5, 266.4, 266.6])
    dec = jnp.array([-28.9, -29.2, -29.0])
    pointing_index = jnp.array([0, 1, 0])
    detector_index = jnp.array([0, 1, 1])

    tel_ra, tel_dec, tel_pa, scale = telescope.pointing.take(pointing_index)
    ideal = telescope.optics.projection(
        tel_ra, tel_dec, tel_pa, ra, dec, scale)
    focal = ideal + telescope.optics.distortion(ideal)
    expected = telescope.detector(focal, detector_index)

    assert telescope.focal_plane(
        ra, dec, pointing_index) == approx(focal)
    assert telescope(
        ra, dec, pointing_index, detector_index) == approx(expected)
    assert eqx.filter_jit(telescope)(
        ra, dec, pointing_index, detector_index) == approx(expected)


def test_telescope_zodiax_update():
    telescope = generate_telescope()
    path = 'optics.distortion.coeff_x'
    updated = telescope.set(path, jnp.zeros(18))

    assert telescope.get(path) == approx(jnp.linspace(0.0, 0.1, 18))
    assert updated.get(path) == approx(jnp.zeros(18))


def test_telescope_gradient():
    telescope = generate_telescope()
    ra = jnp.array([266.5, 266.4])
    dec = jnp.array([-28.9, -29.2])
    pointing_index = jnp.array([0, 1])
    detector_index = jnp.array([0, 1])

    def loss(model):
        value = model(ra, dec, pointing_index, detector_index)
        return jnp.sum(value**2)

    gradient = eqx.filter_grad(loss)(telescope)

    assert jnp.isfinite(gradient.pointing.ra).all()
    assert jnp.isfinite(gradient.optics.distortion.coeff_x).all()
    assert jnp.isfinite(gradient.detector.offset).all()


def test_telescope_validation():
    telescope = generate_telescope()

    with raises(ValueError, match='same shape'):
        telescope.focal_plane([1.0], [2.0, 3.0], [0])
    with raises(ValueError, match='contain integers'):
        telescope.focal_plane([1.0], [2.0], [0.0])
    with raises(TypeError, match='Pointing'):
        Telescope(object(), telescope.optics, telescope.detector)
