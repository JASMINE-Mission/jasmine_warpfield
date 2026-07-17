#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Simple gnomonic system with Legendre distortion '''

import numpy as np

from ..distortion import LegendreDistortion
from .gnomonic import _get_gnomonic


__all__ = ['get_simple_legendre']


def get_simple_legendre(coeff_x=None, coeff_y=None, simulator=False):
    ''' Return the simple gnomonic system with Legendre distortion

    Coefficient arrays have shape ``(18,)`` and default to zero.
    '''
    if coeff_x is None:
        coeff_x = np.zeros(18)
    if coeff_y is None:
        coeff_y = np.zeros(18)
    distortion = LegendreDistortion(coeff_x, coeff_y)
    return _get_gnomonic(distortion, simulator)
