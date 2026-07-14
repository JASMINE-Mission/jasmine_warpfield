#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differentiable telescope model for astrometric analysis '''

import jax.numpy as jnp
import zodiax as zdx

from .detector import Detector
from .optics import Optics
from .pointing import Pointing


__all__ = ['Telescope']


class Telescope(zdx.Base):
    ''' Composition of pointing, optics, and detector models

    Attributes:
        pointing: Pointing parameters for each exposure.
        optics: Projection and focal-plane distortion model.
        detector: Detector geometry model.
    '''

    pointing: Pointing
    optics: Optics
    detector: Detector

    def __init__(self, pointing, optics, detector):
        if not isinstance(pointing, Pointing):
            raise TypeError('`pointing` should be a Pointing instance.')
        if not isinstance(optics, Optics):
            raise TypeError('`optics` should be an Optics instance.')
        if not isinstance(detector, Detector):
            raise TypeError('`detector` should be a Detector instance.')

        self.pointing = pointing
        self.optics = optics
        self.detector = detector

    @staticmethod
    def _validate_sky(ra, dec, pointing_index):
        ra = jnp.asarray(ra, dtype=float)
        dec = jnp.asarray(dec, dtype=float)
        pointing_index = jnp.asarray(pointing_index)

        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        if dec.shape != ra.shape:
            raise ValueError('`dec` should have the same shape as `ra`.')
        if pointing_index.shape != ra.shape:
            raise ValueError(
                '`pointing_index` should have the same shape as `ra`.')
        if not jnp.issubdtype(pointing_index.dtype, jnp.integer):
            raise ValueError('`pointing_index` should contain integers.')

        return ra, dec, pointing_index

    def focal_plane(self, ra, dec, pointing_index):
        ''' Map sky coordinates onto the distorted focal plane '''
        ra, dec, pointing_index = self._validate_sky(
            ra, dec, pointing_index)
        tel_ra, tel_dec, tel_pa, scale = self.pointing.take(pointing_index)
        return self.optics(tel_ra, tel_dec, tel_pa, ra, dec, scale)

    def __call__(self, ra, dec, pointing_index, detector_index):
        ''' Map sky coordinates onto detector coordinates '''
        xy = self.focal_plane(ra, dec, pointing_index)
        return self.detector(xy, detector_index)
