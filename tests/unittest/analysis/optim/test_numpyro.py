#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import jax.random as random
from numpyro import handlers
import numpyro.distributions as dist
from pytest import approx, raises

from warpfield.analysis.optim.numpyro import apply_sample, build_model

from .util import OFFSET_PATH, generate_problem


def test_build_numpyro_model():
    astrometry, measurement = generate_problem()
    priors = {
        OFFSET_PATH: dist.Normal(jnp.zeros(2), jnp.ones(2)),
    }
    model = build_model(astrometry, measurement, priors)
    trace = handlers.trace(
        handlers.seed(model, random.PRNGKey(0))).get_trace()

    assert OFFSET_PATH in trace
    assert trace[OFFSET_PATH]['value'].shape == (2,)
    assert trace['predicted_xy']['value'].shape == (1, 2)
    assert trace['xy']['is_observed']
    assert trace['xy']['value'] == approx(measurement.xy)


def test_apply_sample():
    astrometry, _ = generate_problem()
    sample = {OFFSET_PATH: jnp.array([0.2, -0.1])}
    updated = apply_sample(astrometry, sample, [OFFSET_PATH])

    assert astrometry.get(OFFSET_PATH) == approx([0.0, 0.0])
    assert updated.get(OFFSET_PATH) == approx([0.2, -0.1])


def test_numpyro_model_validation():
    astrometry, measurement = generate_problem()
    measurement = measurement.set('uncertainty', None)
    prior = {OFFSET_PATH: dist.Normal(jnp.zeros(2), jnp.ones(2))}

    with raises(ValueError, match='uncertainty'):
        build_model(astrometry, measurement, prior)
    with raises(ValueError, match='broadcastable'):
        build_model(astrometry, measurement, prior, jnp.ones(3))
    with raises(ValueError, match='at least one'):
        build_model(astrometry, measurement, {}, 1.0)
