#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp

from warpfield import (
    Astrometry,
    Detector,
    Exposure,
    Measurement,
    Optics,
    Pointing,
    SourceCatalog,
    Telescope,
)
from warpfield.calibration import IdentityCalibration
from warpfield.distortion import IdentityDistortion
from warpfield.projection import GnomonicProjection


OFFSET_PATH = 'telescope.detectors.0.offset'


def generate_problem():
    source = SourceCatalog([0.1], [0.2])
    pointing = Pointing([0.0], [0.0], [0.0])
    exposure = Exposure(pointing, IdentityCalibration())
    optics = Optics(
        GnomonicProjection(), IdentityDistortion(), plate_scale=[1.0, 1.0])
    detector = Detector(0.0, [0.0, 0.0], [1.0, 1.0])
    astrometry = Astrometry(
        source,
        Telescope(optics, (detector,)),
        exposure,
    )

    indices = jnp.array([0])
    template = Measurement(
        jnp.zeros((1, 2)), indices, indices, indices, [[1.0, 1.0]])
    target = astrometry.set(OFFSET_PATH, jnp.array([0.2, -0.1]))
    measurement = Measurement(
        target(template), indices, indices, indices, [[1.0, 1.0]])
    return astrometry, measurement
