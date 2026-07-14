#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Telescope pointing parameters for astrometric analysis '''

from jax import Array
import jax.numpy as jnp
import zodiax as zdx


__all__ = ['Pointing']


class Pointing(zdx.Base):
    ''' Telescope pointings represented as a PyTree

    Attributes:
        ra: Right ascensions in degrees with shape ``(N_pointing,)``.
        dec: Declinations in degrees with shape ``(N_pointing,)``.
        position_angle: Position angles in degrees with shape
            ``(N_pointing,)``.
        scale: Focal-plane scales in mm/degree with shape
            ``(N_pointing, 2)``.
    '''

    ra: Array
    dec: Array
    position_angle: Array
    scale: Array

    def __init__(self, ra, dec, position_angle, scale):
        ra = jnp.asarray(ra, dtype=float)
        dec = jnp.asarray(dec, dtype=float)
        position_angle = jnp.asarray(position_angle, dtype=float)
        scale = jnp.asarray(scale, dtype=float)

        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        if dec.shape != ra.shape:
            raise ValueError('`dec` should have the same shape as `ra`.')
        if position_angle.shape != ra.shape:
            raise ValueError(
                '`position_angle` should have the same shape as `ra`.')
        if scale.shape != (ra.shape[0], 2):
            raise ValueError('`scale` should have shape (N_pointing, 2).')

        self.ra = ra
        self.dec = dec
        self.position_angle = position_angle
        self.scale = scale

    def __len__(self):
        return self.ra.shape[0]

    def take(self, index):
        ''' Select pointing parameters using an integer index array '''
        return (
            self.ra[index],
            self.dec[index],
            self.position_angle[index],
            self.scale[index],
        )
