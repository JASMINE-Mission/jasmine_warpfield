#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Detector geometry for astrometric analysis '''

import equinox as eqx
from jax import Array
import jax.numpy as jnp
import zodiax as zdx

from .utils import _affine_transform


__all__ = ['Detector']


class Detector(zdx.Base):
    ''' Geometry of one detector represented as a PyTree

    Attributes:
        rotation: Rotation angle in degrees represented as a scalar.
        offset: Focal-plane offset in mm with shape ``(2,)``.
        pixel_scale: Physical pixel size in mm/pixel with shape ``(2,)``.
        shape: Detector dimensions in pixels as ``(NAXIS1, NAXIS2)``.
    '''

    rotation: Array
    offset: Array
    pixel_scale: Array
    shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
            self, rotation, offset, pixel_scale, shape=(1024, 1024)):
        rotation = jnp.asarray(rotation, dtype=float)
        offset = jnp.asarray(offset, dtype=float)
        pixel_scale = jnp.asarray(pixel_scale, dtype=float)

        if rotation.ndim != 0:
            raise ValueError('`rotation` should be a scalar.')
        if offset.shape != (2,):
            raise ValueError('`offset` should have shape (2,).')
        if pixel_scale.shape != (2,):
            raise ValueError('`pixel_scale` should have shape (2,).')
        if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or not all(
                    isinstance(size, int) and not isinstance(size, bool)
                    for size in shape)):
            raise TypeError('`shape` should be a tuple of two integers.')
        if any(size <= 0 for size in shape):
            raise ValueError('Detector dimensions should be positive.')

        self.rotation = rotation
        self.offset = offset
        self.pixel_scale = pixel_scale
        self.shape = shape

    def __call__(self, xy):
        ''' Transform focal-plane coordinates onto this detector '''
        xy = jnp.asarray(xy)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('`xy` should have shape (N_coordinate, 2).')

        size = xy.shape[0]
        rotation = jnp.broadcast_to(self.rotation, (size,))
        offset = jnp.broadcast_to(self.offset, (size, 2))
        pixel_scale = jnp.broadcast_to(self.pixel_scale, (size, 2))
        return _affine_transform(xy, rotation, offset, pixel_scale)


def _apply_detectors(detectors, xy, detector_index):
    ''' Transform coordinates using the selected detectors '''
    xy = jnp.asarray(xy)
    detector_index = jnp.asarray(detector_index)

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('`xy` should have shape (N_observation, 2).')
    if detector_index.ndim != 1:
        raise ValueError(
            '`detector_index` should be a one-dimensional array.')
    if detector_index.shape[0] != xy.shape[0]:
        raise ValueError(
            '`xy` and `detector_index` should have the same length.')
    if not jnp.issubdtype(detector_index.dtype, jnp.integer):
        raise ValueError('`detector_index` should contain integers.')

    rotation = jnp.stack([detector.rotation for detector in detectors])
    offset = jnp.stack([detector.offset for detector in detectors])
    pixel_scale = jnp.stack([
        detector.pixel_scale for detector in detectors])
    return _affine_transform(
        xy,
        rotation[detector_index],
        offset[detector_index],
        pixel_scale[detector_index],
    )
