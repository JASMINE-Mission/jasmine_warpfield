#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from pytest import approx, raises

from warpfield.analysis.optim import get_parameters, set_parameters

from .util import OFFSET_PATH, generate_problem


def test_get_and_set_parameters():
    astrometry, _ = generate_problem()
    parameters = get_parameters(astrometry, [OFFSET_PATH])
    updated = set_parameters(
        astrometry, {OFFSET_PATH: jnp.array([0.2, -0.1])})

    assert list(parameters) == [OFFSET_PATH]
    assert parameters[OFFSET_PATH] == approx([0.0, 0.0])
    assert astrometry.get(OFFSET_PATH) == approx([0.0, 0.0])
    assert updated.get(OFFSET_PATH) == approx([0.2, -0.1])


def test_parameter_validation():
    astrometry, _ = generate_problem()

    with raises(TypeError, match='zodiax'):
        get_parameters(object(), [OFFSET_PATH])
    with raises(ValueError, match='at least one'):
        get_parameters(astrometry, [])
    with raises(TypeError, match='mapping'):
        set_parameters(astrometry, [])
    with raises(ValueError, match='at least one'):
        set_parameters(astrometry, {})
