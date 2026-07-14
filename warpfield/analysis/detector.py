#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Detector parameters for astrometric analysis '''

from jax import Array
import jax.numpy as jnp
import zodiax as zdx

from .utils import _affine_transform


__all__ = ['Detector']


class Detector(zdx.Base):
    ''' Detector geometries represented as a PyTree

    Attributes:
        rotation: Rotation angles in degrees with shape ``(N_detector,)``.
        offset: Focal-plane offsets in mm with shape ``(N_detector, 2)``.
        pixel_scale: Physical pixel sizes in mm/pixel with shape
            ``(N_detector, 2)``.
    '''

    rotation: Array
    offset: Array
    pixel_scale: Array

    def __init__(self, rotation, offset, pixel_scale):
        rotation = jnp.asarray(rotation, dtype=float)
        offset = jnp.asarray(offset, dtype=float)
        pixel_scale = jnp.asarray(pixel_scale, dtype=float)

        if rotation.ndim != 1:
            raise ValueError('`rotation` should be a one-dimensional array.')
        if offset.shape != (rotation.shape[0], 2):
            raise ValueError('`offset` should have shape (N_detector, 2).')
        if pixel_scale.shape != (rotation.shape[0], 2):
            raise ValueError(
                '`pixel_scale` should have shape (N_detector, 2).')

        self.rotation = rotation
        self.offset = offset
        self.pixel_scale = pixel_scale

    def __len__(self):
        return self.rotation.shape[0]

    def take(self, index):
        ''' Select detector parameters using an integer index array '''
        return (
            self.rotation[index],
            self.offset[index],
            self.pixel_scale[index],
        )

    def __call__(self, xy, detector_index):
        ''' Transform focal-plane coordinates into detector coordinates '''
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

        rotation, offset, pixel_scale = self.take(detector_index)
        return _affine_transform(xy, rotation, offset, pixel_scale)
