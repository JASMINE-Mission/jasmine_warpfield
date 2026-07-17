#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Projection of the spherical coordinate onto the focal plane '''

from .base import Projection
from .equidistant import EquidistantProjection
from .gnomonic import GnomonicProjection
from .orthographic import OrthographicProjection


__all__ = [
    'EquidistantProjection',
    'GnomonicProjection',
    'OrthographicProjection',
    'Projection',
]
