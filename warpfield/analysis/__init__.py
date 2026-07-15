#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differentiable astrometric analysis '''

import jax

from .source import SourceCatalog
from .pointing import Pointing
from .detector import Detector
from .exposure import Exposure
from .measurement import Measurement
from .optics import Optics
from .telescope import Telescope
from .astrometry import Astrometry


jax.config.update('jax_enable_x64', True)


__all__ = [
    'SourceCatalog',
    'Pointing',
    'Detector',
    'Exposure',
    'Measurement',
    'Optics',
    'Telescope',
    'Astrometry',
]
