#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Differentiable astrometric analysis with JAX '''

import jax

from .catalog import AstrometricCatalog, SourceCatalog
from .pointing import Pointing
from .detector import Detector
from .exposure import Exposure
from .measurement import Measurement
from .optics import Optics
from .telescope import Telescope
from .simulator import Simulator
from .astrometry import Astrometry
from .version import version as __version__


jax.config.update('jax_enable_x64', True)


__all__ = [
    '__version__',
    'AstrometricCatalog',
    'SourceCatalog',
    'Pointing',
    'Detector',
    'Exposure',
    'Measurement',
    'Optics',
    'Telescope',
    'Simulator',
    'Astrometry',
]
