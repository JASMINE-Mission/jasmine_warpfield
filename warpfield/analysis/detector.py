#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Detector geometry for astrometric analysis '''

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
    '''

    rotation: Array
    offset: Array
    pixel_scale: Array

    def __init__(self, rotation, offset, pixel_scale):
        rotation = jnp.asarray(rotation, dtype=float)
        offset = jnp.asarray(offset, dtype=float)
        pixel_scale = jnp.asarray(pixel_scale, dtype=float)

        if rotation.ndim != 0:
            raise ValueError('`rotation` should be a scalar.')
        if offset.shape != (2,):
            raise ValueError('`offset` should have shape (2,).')
        if pixel_scale.shape != (2,):
            raise ValueError('`pixel_scale` should have shape (2,).')

        self.rotation = rotation
        self.offset = offset
        self.pixel_scale = pixel_scale

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
