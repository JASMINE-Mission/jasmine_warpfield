#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Optax adapter for astrometric least-squares optimisation '''

import jax
import jax.numpy as jnp
import optax as ox

from .parameters import get_parameters, set_parameters


__all__ = ['initialize', 'least_squares', 'step']


def initialize(astrometry, paths, optimizer):
    ''' Extract trainable parameters and initialize an Optax state '''
    parameters = get_parameters(astrometry, paths)
    return parameters, optimizer.init(parameters)


def least_squares(parameters, astrometry, measurement):
    ''' Calculate a weighted least-squares objective '''
    current = set_parameters(astrometry, parameters)
    residual = current.residual(measurement)
    if measurement.uncertainty is not None:
        residual = residual / measurement.uncertainty
    return 0.5 * jnp.sum(residual**2)


def step(parameters, state, optimizer, astrometry, measurement):
    ''' Apply one Optax update and return parameters, state, and loss '''
    value, gradient = jax.value_and_grad(least_squares)(
        parameters, astrometry, measurement)
    updates, state = optimizer.update(gradient, state, parameters)
    parameters = ox.apply_updates(parameters, updates)
    return parameters, state, value
