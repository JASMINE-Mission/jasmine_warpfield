#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differentiable telescope optics model '''

import zodiax as zdx

from .distortion import Distortion
from .projection import Projection


__all__ = ['Optics']


class Optics(zdx.Base):
    ''' Composition of a sky projection and a distortion model

    Attributes:
        projection: Model that projects sky coordinates onto the focal plane.
        distortion: Model that returns focal-plane coordinate displacements.
    '''

    projection: Projection
    distortion: Distortion

    def __init__(self, projection, distortion):
        if not isinstance(projection, Projection):
            raise TypeError('`projection` should be a Projection instance.')
        if not isinstance(distortion, Distortion):
            raise TypeError('`distortion` should be a Distortion instance.')

        self.projection = projection
        self.distortion = distortion

    def project(self, tel_ra, tel_dec, tel_pa, ra, dec, scale):
        ''' Project sky coordinates onto the ideal focal plane '''
        return self.projection(tel_ra, tel_dec, tel_pa, ra, dec, scale)

    def distort(self, xy):
        ''' Apply coordinate displacements to ideal focal-plane positions '''
        return xy + self.distortion(xy)

    def __call__(self, tel_ra, tel_dec, tel_pa, ra, dec, scale):
        ''' Project sky coordinates and apply the configured distortion '''
        return self.distort(
            self.project(tel_ra, tel_dec, tel_pa, ra, dec, scale))
