#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
from pytest import approx, raises

from warpfield.distortion import (
    IdentityDistortion,
    LegendreDistortion,
    SIPDistortion,
)
from warpfield.model import (
    get_gnomonic,
    get_simple_legendre,
    get_simple_sip,
)
from warpfield.simulator import Simulator
from warpfield.telescope import Telescope


def test_get_gnomonic():
    model = get_gnomonic()
    detector = model.detectors[0]

    assert type(model) is Telescope
    assert isinstance(model.optics.distortion, IdentityDistortion)
    assert model.optics.plate_scale == approx([36.0, 36.0])
    assert model.optics.imaging_radius == approx(
        np.sqrt(2) * 20.48
    )
    assert len(model.detectors) == 1
    assert detector.shape == (4096, 4096)
    assert detector.pixel_scale == approx([0.01, 0.01])
    assert detector.offset == approx([0.0, 0.0])
    assert float(detector.rotation) == approx(0.0)


def test_get_gnomonic_angular_scale():
    model = get_gnomonic()
    detector_xy = model(
        tel_ra=jnp.array([0.0]),
        tel_dec=jnp.array([0.0]),
        tel_pa=jnp.array([0.0]),
        ra=jnp.array([1 / 3600]),
        dec=jnp.array([0.0]),
        scale_factor=jnp.ones((1, 1)),
        detector_index=jnp.array([0]),
    )

    assert detector_xy == approx(
        np.array([[2047.0, 2048.0]]),
        abs=1e-7,
    )


def test_get_gnomonic_simulator():
    model = get_gnomonic(simulator=True)

    assert isinstance(model, Simulator)


def test_get_simple_sip():
    coeff_x = np.linspace(0.0, 1.0, 18)
    coeff_y = np.linspace(1.0, 0.0, 18)
    model = get_simple_sip(coeff_x, coeff_y)

    assert isinstance(model.optics.distortion, SIPDistortion)
    assert model.optics.distortion.coeff_x == approx(coeff_x)
    assert model.optics.distortion.coeff_y == approx(coeff_y)

    default = get_simple_sip()
    assert default.optics.distortion.coeff_x == approx(np.zeros(18))
    assert default.optics.distortion.coeff_y == approx(np.zeros(18))


def test_get_simple_legendre():
    coeff_x = np.linspace(0.0, 1.0, 18)
    coeff_y = np.linspace(1.0, 0.0, 18)
    model = get_simple_legendre(coeff_x, coeff_y)

    assert isinstance(model.optics.distortion, LegendreDistortion)
    assert model.optics.distortion.coeff_x == approx(coeff_x)
    assert model.optics.distortion.coeff_y == approx(coeff_y)

    default = get_simple_legendre()
    assert default.optics.distortion.coeff_x == approx(np.zeros(18))
    assert default.optics.distortion.coeff_y == approx(np.zeros(18))


def test_simple_model_validation():
    with raises(TypeError, match='boolean'):
        get_gnomonic(simulator=1)
    with raises(ValueError, match='coeff_x'):
        get_simple_sip(np.zeros(17), np.zeros(18))
    with raises(ValueError, match='coeff_y'):
        get_simple_legendre(np.zeros(18), np.zeros(17))
