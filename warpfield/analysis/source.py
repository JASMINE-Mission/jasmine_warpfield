#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Source catalog parameters for astrometric analysis '''

from jax import Array
import jax.numpy as jnp
import zodiax as zdx


class SourceCatalog(zdx.Base):
    ''' Celestial source positions represented as a PyTree

    Attributes:
        ra: Right ascensions in degrees with shape ``(N_source,)``.
        dec: Declinations in degrees with shape ``(N_source,)``.
    '''

    ra: Array
    dec: Array

    def __init__(self, ra, dec):
        ra = jnp.asarray(ra, dtype=float)
        dec = jnp.asarray(dec, dtype=float)

        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        if dec.ndim != 1:
            raise ValueError('`dec` should be a one-dimensional array.')
        if ra.shape != dec.shape:
            raise ValueError('`ra` and `dec` should have the same shape.')

        self.ra = ra
        self.dec = dec

    def __len__(self):
        return self.ra.shape[0]

    def take(self, index):
        ''' Select source positions using an integer index array '''
        return self.ra[index], self.dec[index]


__all__ = ['SourceCatalog']
