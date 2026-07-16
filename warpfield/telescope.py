#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differentiable telescope model for astrometric analysis '''

import jax.numpy as jnp
import zodiax as zdx

from .detector import Detector, _apply_detectors
from .optics import Optics


__all__ = ['Telescope']


class Telescope(zdx.Base):
    ''' Composition of optics and detector models

    Attributes:
        optics: Projection and focal-plane distortion model.
        detectors: Detector geometry models.
    '''

    optics: Optics
    detectors: tuple[Detector, ...]

    def __init__(self, optics, detectors):
        if not isinstance(optics, Optics):
            raise TypeError('`optics` should be an Optics instance.')
        if not isinstance(detectors, tuple):
            raise TypeError('`detectors` should be a tuple of Detector.')
        if len(detectors) == 0:
            raise ValueError(
                '`detectors` should contain at least one Detector.')
        if not all(isinstance(detector, Detector) for detector in detectors):
            raise TypeError('`detectors` should contain only Detector.')

        self.optics = optics
        self.detectors = detectors

    @staticmethod
    def _validate_sky(tel_ra, tel_dec, tel_pa, ra, dec):
        tel_ra = jnp.asarray(tel_ra, dtype=float)
        tel_dec = jnp.asarray(tel_dec, dtype=float)
        tel_pa = jnp.asarray(tel_pa, dtype=float)
        ra = jnp.asarray(ra, dtype=float)
        dec = jnp.asarray(dec, dtype=float)

        if ra.ndim != 1:
            raise ValueError('`ra` should be a one-dimensional array.')
        if dec.shape != ra.shape:
            raise ValueError('`dec` should have the same shape as `ra`.')
        if tel_ra.shape != ra.shape:
            raise ValueError('`tel_ra` should have the same shape as `ra`.')
        if tel_dec.shape != ra.shape:
            raise ValueError('`tel_dec` should have the same shape as `ra`.')
        if tel_pa.shape != ra.shape:
            raise ValueError(
                '`tel_pa` should have the same shape as `ra`.')

        return tel_ra, tel_dec, tel_pa, ra, dec

    def focal_plane(
            self, tel_ra, tel_dec, tel_pa, ra, dec, scale_factor):
        ''' Map sky coordinates onto the distorted focal plane '''
        tel_ra, tel_dec, tel_pa, ra, dec = self._validate_sky(
            tel_ra, tel_dec, tel_pa, ra, dec)
        return self.optics(
            tel_ra, tel_dec, tel_pa, ra, dec, scale_factor)

    def __call__(
            self, tel_ra, tel_dec, tel_pa, ra, dec, scale_factor,
            detector_index):
        ''' Map sky coordinates onto detector coordinates '''
        xy = self.focal_plane(
            tel_ra, tel_dec, tel_pa, ra, dec, scale_factor)
        return _apply_detectors(self.detectors, xy, detector_index)
