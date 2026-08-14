#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numerical utilities for astrometric analysis"""

from astropy.units import Quantity
import astropy.units as u
from jax import vmap
import jax.numpy as jnp
import numpy as np


__all__ = ['plate_scale']


def plate_scale(focal_length):
    """Calculate an isotropic focal-plane scale from a focal length

    Arguments:
        focal_length: Scalar Astropy length Quantity.

    Returns:
        Plate scales along both focal-plane axes in mm/degree with shape
        ``(2,)``.
    """
    if not isinstance(focal_length, Quantity):
        raise TypeError('`focal_length` should be an Astropy Quantity.')
    try:
        focal_length = focal_length.to_value(u.mm)
    except u.UnitConversionError as error:
        raise ValueError('`focal_length` should have length units.') from error
    if np.ndim(focal_length) != 0:
        raise ValueError('`focal_length` should be a scalar.')
    if not np.isfinite(focal_length) or focal_length <= 0:
        raise ValueError('`focal_length` should be finite and positive.')

    scale = focal_length * np.pi / 180
    return jnp.full((2,), scale)


def _degree_to_radian(theta):
    """Convert degrees to radians"""
    return theta * jnp.pi / 180.0


def _rotation_matrix(theta):
    """Calculate a two-dimensional rotation matrix"""
    values = [
        jnp.cos(theta),
        -jnp.sin(theta),
        jnp.sin(theta),
        jnp.cos(theta),
    ]
    return jnp.array(values).reshape((2, 2))


def _affine_transform_one(xy, rotation, offset, pixel_scale):
    """Transform one focal-plane coordinate into detector coordinates"""
    rotation = _degree_to_radian(rotation)
    return _rotation_matrix(rotation) @ (xy - offset).T / pixel_scale


_affine_transform = vmap(_affine_transform_one, (0, 0, 0, 0), 0)
