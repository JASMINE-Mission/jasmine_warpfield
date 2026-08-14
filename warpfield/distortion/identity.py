#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Identity distortion model"""

import jax.numpy as jnp

from .base import Distortion


__all__ = ['IdentityDistortion']


class IdentityDistortion(Distortion):
    """Distortion model that returns zero coordinate displacements"""

    def __call__(self, xy):
        xy = jnp.asarray(xy)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('`xy` should have shape (N_coordinate, 2).')
        return jnp.zeros_like(xy)
