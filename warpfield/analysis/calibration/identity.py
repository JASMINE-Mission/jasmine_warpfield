#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Identity exposure calibration '''

import jax.numpy as jnp

from .base import Calibration


__all__ = ['IdentityCalibration']


class IdentityCalibration(Calibration):
    ''' Calibration that leaves the nominal plate scale unchanged '''

    def scale_factor(self, exposure_index):
        exposure_index = jnp.asarray(exposure_index)
        return jnp.ones((exposure_index.shape[0], 1))
