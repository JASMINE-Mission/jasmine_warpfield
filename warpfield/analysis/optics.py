#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differentiable telescope optics model '''

from jax import Array
import jax.numpy as jnp
import zodiax as zdx

from .distortion import Distortion
from .projection import Projection


__all__ = ['Optics']


class Optics(zdx.Base):
    ''' Composition of a sky projection and a distortion model

    Attributes:
        projection: Model that projects sky coordinates onto the focal plane.
        distortion: Model that returns focal-plane coordinate displacements.
        plate_scale: Nominal focal-plane scale in mm/degree with shape
            ``(2,)``.
    '''

    projection: Projection
    distortion: Distortion
    plate_scale: Array

    def __init__(self, projection, distortion, plate_scale):
        if not isinstance(projection, Projection):
            raise TypeError('`projection` should be a Projection instance.')
        if not isinstance(distortion, Distortion):
            raise TypeError('`distortion` should be a Distortion instance.')
        plate_scale = jnp.asarray(plate_scale, dtype=float)
        if plate_scale.shape != (2,):
            raise ValueError('`plate_scale` should have shape (2,).')

        self.projection = projection
        self.distortion = distortion
        self.plate_scale = plate_scale

    def project(self, tel_ra, tel_dec, tel_pa, ra, dec, scale_factor):
        ''' Project sky coordinates onto the ideal focal plane '''
        scale_factor = jnp.asarray(scale_factor)
        if scale_factor.ndim != 2 or scale_factor.shape[1] not in (1, 2):
            raise ValueError(
                '`scale_factor` should have shape (N_coordinate, 1) or '
                '(N_coordinate, 2).')
        if scale_factor.shape[0] != jnp.asarray(ra).shape[0]:
            raise ValueError(
                '`scale_factor` and coordinates should have the same length.')
        scale = self.plate_scale * scale_factor
        return self.projection(tel_ra, tel_dec, tel_pa, ra, dec, scale)

    def distort(self, xy):
        ''' Apply coordinate displacements to ideal focal-plane positions '''
        return xy + self.distortion(xy)

    def __call__(self, tel_ra, tel_dec, tel_pa, ra, dec, scale_factor):
        ''' Project sky coordinates and apply the configured distortion '''
        return self.distort(
            self.project(tel_ra, tel_dec, tel_pa, ra, dec, scale_factor))
