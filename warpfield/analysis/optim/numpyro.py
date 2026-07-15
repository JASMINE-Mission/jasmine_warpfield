#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' NumPyro adapter for probabilistic astrometric inference '''

from collections.abc import Mapping

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from ..astrometry import Astrometry
from ..measurement import Measurement
from .parameters import set_parameters


__all__ = ['apply_sample', 'build_model']


def _prepare_uncertainty(measurement, uncertainty):
    if uncertainty is None:
        uncertainty = measurement.uncertainty
    if uncertainty is None:
        raise ValueError(
            'Measurement uncertainty or `uncertainty` should be provided.')
    uncertainty = jnp.asarray(uncertainty, dtype=float)
    try:
        return jnp.broadcast_to(uncertainty, measurement.xy.shape)
    except ValueError as error:
        raise ValueError(
            '`uncertainty` should be broadcastable to measurement.xy.') \
            from error


def build_model(astrometry, measurement, priors, uncertainty=None):
    ''' Build a NumPyro model using an Astrometry forward calculation '''
    if not isinstance(astrometry, Astrometry):
        raise TypeError('`astrometry` should be an Astrometry instance.')
    if not isinstance(measurement, Measurement):
        raise TypeError('`measurement` should be a Measurement instance.')
    if not isinstance(priors, Mapping):
        raise TypeError('`priors` should be a mapping.')
    if len(priors) == 0:
        raise ValueError('`priors` should contain at least one distribution.')

    uncertainty = _prepare_uncertainty(measurement, uncertainty)

    def model():
        parameters = {
            path: numpyro.sample(path, prior)
            for path, prior in priors.items()
        }
        current = set_parameters(astrometry, parameters)
        predicted = numpyro.deterministic(
            'predicted_xy', current(measurement))

        with numpyro.plate('measurement_plate', len(measurement)):
            numpyro.sample(
                'xy',
                dist.Normal(predicted, uncertainty).to_event(1),
                obs=measurement.xy,
            )

    return model


def apply_sample(astrometry, sample, paths):
    ''' Apply one posterior sample to an Astrometry PyTree '''
    parameters = {path: sample[path] for path in paths}
    return set_parameters(astrometry, parameters)
