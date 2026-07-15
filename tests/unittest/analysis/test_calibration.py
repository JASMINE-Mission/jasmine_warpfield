#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis.calibration import (
    IdentityCalibration,
    ScaleCalibration,
)


def test_identity_calibration():
    calibration = IdentityCalibration()
    factor = calibration.scale_factor(jnp.array([1, 0, 1]))

    assert isinstance(calibration, zdx.Base)
    assert calibration.num_exposure is None
    assert factor == approx(jnp.ones((3, 1)))


def test_scale_calibration():
    coefficient = jnp.log(jnp.array([1.0, 1.1]))
    calibration = ScaleCalibration(coefficient)
    factor = calibration.scale_factor(jnp.array([1, 0, 1]))

    assert isinstance(calibration, zdx.Base)
    assert calibration.num_exposure == 2
    assert factor == approx(jnp.array([[1.1], [1.0], [1.1]]))
    assert len(jax.tree_util.tree_leaves(calibration)) == 1


def test_scale_calibration_zodiax_update():
    calibration = ScaleCalibration([0.0, 0.1])
    updated = calibration.set('coefficient', jnp.array([0.2, 0.3]))

    assert calibration.get('coefficient') == approx([0.0, 0.1])
    assert updated.get('coefficient') == approx([0.2, 0.3])


def test_scale_calibration_validation():
    with raises(ValueError, match='one-dimensional'):
        ScaleCalibration([[0.0]])
