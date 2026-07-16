#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Projection of the spherical coordinate onto the focal plane '''

from .base import Projection
from .gnomonic import GnomonicProjection
from .equidistant import EquidistantProjection


__all__ = [
    'Projection',
    'GnomonicProjection',
    'EquidistantProjection',
]
