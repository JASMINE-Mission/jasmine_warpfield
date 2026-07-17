#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax.numpy as jnp

from warpfield import (
    Astrometry,
    Detector,
    Exposure,
    Measurement,
    Pointing,
    SourceCatalog,
    Telescope,
)
from warpfield.calibration import IdentityCalibration
from warpfield.projection import GnomonicProjection


OFFSET_PATH = 'telescope.detectors.0.offset'


def generate_problem():
    source = SourceCatalog([0.1], [0.2])
    pointing = Pointing([0.0], [0.0], [0.0])
    exposure = Exposure(pointing, IdentityCalibration())
    detector = Detector(0.0, [0.0, 0.0], [1.0, 1.0])
    astrometry = Astrometry(
        source,
        Telescope(GnomonicProjection(), [1.0, 1.0], (detector,)),
        exposure,
    )

    indices = jnp.array([0])
    template = Measurement(
        jnp.zeros((1, 2)), indices, indices, indices, [[1.0, 1.0]])
    target = astrometry.set(OFFSET_PATH, jnp.array([0.2, -0.1]))
    measurement = Measurement(
        target(template), indices, indices, indices, [[1.0, 1.0]])
    return astrometry, measurement
