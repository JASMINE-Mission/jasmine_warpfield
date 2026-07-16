#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Exposure-dependent plate-scale calibration '''

from jax import Array
import jax.numpy as jnp

from .base import Calibration


__all__ = ['ScaleCalibration']


class ScaleCalibration(Calibration):
    ''' Isotropic plate-scale variations for each exposure

    Attributes:
        coefficient: Logarithmic scale factors with shape ``(N_exposure,)``.
            A value of zero leaves the nominal plate scale unchanged.
    '''

    coefficient: Array

    def __init__(self, coefficient):
        coefficient = jnp.asarray(coefficient, dtype=float)
        if coefficient.ndim != 1:
            raise ValueError(
                '`coefficient` should be a one-dimensional array.')
        self.coefficient = coefficient

    def __getitem__(self, index):
        ''' Select coefficients while preserving the collection dimension '''
        return ScaleCalibration(jnp.atleast_1d(self.coefficient[index]))

    @property
    def num_exposure(self):
        return self.coefficient.shape[0]

    def scale_factor(self, exposure_index):
        exposure_index = jnp.asarray(exposure_index)
        return jnp.exp(self.coefficient[exposure_index, None])
