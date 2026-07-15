#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
import jax.numpy as jnp
from pytest import approx, raises

from warpfield.analysis import Detector, Optics, Telescope
from warpfield.analysis.distortion import LegendreDistortion
from warpfield.analysis.projection import GnomonicProjection


def generate_telescope():
    distortion = LegendreDistortion(
        coeff_x=jnp.linspace(0.0, 0.1, 18),
        coeff_y=jnp.linspace(0.1, 0.0, 18),
        plane_scale=10.0,
    )
    optics = Optics(
        GnomonicProjection(), distortion, plate_scale=[2.0, 3.0])
    detectors = (
        Detector(0.0, [0.0, 0.0], [1.0, 1.0]),
        Detector(90.0, [1.0, 2.0], [0.5, 2.0]),
    )
    return Telescope(optics, detectors)


def generate_coordinates():
    return (
        jnp.array([266.4, 266.5, 266.4]),
        jnp.array([-29.0, -29.1, -29.0]),
        jnp.array([0.0, 5.0, 0.0]),
        jnp.array([266.5, 266.4, 266.6]),
        jnp.array([-28.9, -29.2, -29.0]),
        jnp.array([[1.0], [1.1], [0.9]]),
    )


def test_telescope_pipeline():
    telescope = generate_telescope()
    coordinates = generate_coordinates()
    detector_index = jnp.array([0, 1, 1])
    ideal = telescope.optics.projection(
        *coordinates[:-1],
        telescope.optics.plate_scale * coordinates[-1],
    )
    focal = ideal + telescope.optics.distortion(ideal)
    expected = jnp.concatenate([
        telescope.detectors[0](focal[:1]),
        telescope.detectors[1](focal[1:]),
    ])

    assert telescope.focal_plane(*coordinates) == approx(focal)
    assert telescope(*coordinates, detector_index) == approx(expected)
    assert eqx.filter_jit(telescope)(
        *coordinates, detector_index) == approx(expected)


def test_telescope_zodiax_update():
    telescope = generate_telescope()
    path = 'optics.plate_scale'
    updated = telescope.set(path, jnp.array([3.0, 4.0]))

    assert telescope.get(path) == approx([2.0, 3.0])
    assert updated.get(path) == approx([3.0, 4.0])

    path = 'detectors.1.offset'
    updated = telescope.set(path, jnp.zeros(2))

    assert telescope.get(path) == approx([1.0, 2.0])
    assert updated.get(path) == approx(jnp.zeros(2))


def test_telescope_gradient():
    telescope = generate_telescope()
    coordinates = generate_coordinates()
    detector_index = jnp.array([0, 1, 0])

    def loss(model):
        value = model(*coordinates, detector_index)
        return jnp.sum(value**2)

    gradient = eqx.filter_grad(loss)(telescope)

    assert jnp.isfinite(gradient.optics.plate_scale).all()
    assert jnp.isfinite(gradient.optics.distortion.coeff_x).all()
    assert jnp.isfinite(gradient.detectors[0].offset).all()
    assert jnp.isfinite(gradient.detectors[1].offset).all()


def test_telescope_validation():
    telescope = generate_telescope()
    coordinates = generate_coordinates()

    with raises(ValueError, match='same shape'):
        telescope.focal_plane(*coordinates[:3], [1.0], *coordinates[4:])
    with raises(TypeError, match='Optics'):
        Telescope(object(), telescope.detectors)
    with raises(TypeError, match='tuple'):
        Telescope(telescope.optics, [])
    with raises(ValueError, match='at least one'):
        Telescope(telescope.optics, ())
    with raises(TypeError, match='only Detector'):
        Telescope(telescope.optics, (object(),))
