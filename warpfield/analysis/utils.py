#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Internal numerical utilities for astrometric analysis '''

from jax import vmap
import jax.numpy as jnp


__all__ = []


def _degree_to_radian(theta):
    ''' Convert degrees to radians '''
    return theta * jnp.pi / 180.0


def _rotation_matrix(theta):
    ''' Calculate a two-dimensional rotation matrix '''
    values = [
        jnp.cos(theta),
        -jnp.sin(theta),
        jnp.sin(theta),
        jnp.cos(theta),
    ]
    return jnp.array(values).reshape((2, 2))


def _affine_transform_one(xy, rotation, offset, pixel_scale):
    ''' Transform one focal-plane coordinate into detector coordinates '''
    rotation = _degree_to_radian(rotation)
    return _rotation_matrix(rotation) @ (xy - offset).T / pixel_scale


_affine_transform = vmap(_affine_transform_one, (0, 0, 0, 0), 0)
