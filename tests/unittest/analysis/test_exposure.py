#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import Exposure, Pointing
from warpfield.analysis.calibration import (
    IdentityCalibration,
    ScaleCalibration,
)


def generate_pointing():
    return Pointing(
        ra=[10.0, 20.0],
        dec=[-10.0, -20.0],
        position_angle=[1.0, 2.0],
    )


def test_exposure():
    exposure = Exposure(
        generate_pointing(),
        ScaleCalibration(jnp.log(jnp.array([1.0, 1.1]))),
    )
    ra, dec, position_angle, factor = exposure.take(jnp.array([1, 0]))

    assert isinstance(exposure, zdx.Base)
    assert len(exposure) == 2
    assert ra == approx([20.0, 10.0])
    assert dec == approx([-20.0, -10.0])
    assert position_angle == approx([2.0, 1.0])
    assert factor == approx(jnp.array([[1.1], [1.0]]))
    assert len(jax.tree_util.tree_leaves(exposure)) == 4


def test_identity_calibrated_exposure():
    exposure = Exposure(generate_pointing(), IdentityCalibration())

    assert exposure.take(jnp.array([0, 1]))[-1] == approx(
        jnp.ones((2, 1)))


def test_exposure_zodiax_update():
    exposure = Exposure(generate_pointing(), ScaleCalibration([0.0, 0.1]))
    updated = exposure.set(
        'calibration.coefficient', jnp.array([0.2, 0.3]))

    assert exposure.get('calibration.coefficient') == approx([0.0, 0.1])
    assert updated.get('calibration.coefficient') == approx([0.2, 0.3])


def test_exposure_validation():
    pointing = generate_pointing()

    with raises(TypeError, match='Pointing'):
        Exposure(object(), IdentityCalibration())
    with raises(TypeError, match='Calibration'):
        Exposure(pointing, object())
    with raises(ValueError, match='same length'):
        Exposure(pointing, ScaleCalibration([0.0]))
