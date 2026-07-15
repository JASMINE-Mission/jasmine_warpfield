#!/usr/bin/env python
# -*- coding: utf-8 -*-

import equinox as eqx
from pytest import approx
import optax

from warpfield.analysis.optim import set_parameters
from warpfield.analysis.optim.optax import initialize, least_squares, step

from .util import OFFSET_PATH, generate_problem


def test_optax_step():
    astrometry, measurement = generate_problem()
    optimizer = optax.sgd(learning_rate=0.5)
    parameters, state = initialize(astrometry, [OFFSET_PATH], optimizer)
    initial = least_squares(parameters, astrometry, measurement)

    parameters, state, value = eqx.filter_jit(step)(
        parameters, state, optimizer, astrometry, measurement)
    final = least_squares(parameters, astrometry, measurement)
    fitted = set_parameters(astrometry, parameters)

    assert value == approx(initial)
    assert final < initial
    assert fitted.get(OFFSET_PATH) == approx([0.1, -0.05])


def test_unweighted_least_squares():
    astrometry, measurement = generate_problem()
    measurement = measurement.set('uncertainty', None)
    parameters, _ = initialize(
        astrometry, [OFFSET_PATH], optax.sgd(learning_rate=0.1))

    assert least_squares(parameters, astrometry, measurement) > 0.0
